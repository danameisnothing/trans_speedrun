import groq
import srt
import silero_vad
from google import genai

import time
import os
import datetime
import json
import argparse

GEMINI_BASE_PROMPT = """You are a model that is used to fix transcription errors / omissions. You will be given a JSON object with the following format :
```json
{
    "list": ["<segment text 1>", "<segment text 2>", "<segment text 3>", "<etc>"]
}
```

Here are the rules that you need to follow for fixing the given transcriptions :
- TRY to correct words that you sense to have wrong capitalizations and / or does not have enough emphasis, or just flat out wrong. Feel free to correct them, including, but not limited to, by correcting its capitalization, or add more punctuation
- Do NOT apply any censorship!
- For any mentioned names, only correct its capitalization and nothing else
- Do NOT use more than 1 whitespace character after another!
You will respond in the following format :
```json
{
    "response": [
        "<corrected text 1>",
        "<corrected text 2>",
        "<corrected text 3>",
        "<corrected text 4>",
        "<corrected text 5>",
        "<corrected text 6>",
        "<etc>"
    ],
    "original_array_length": <number here>,
    "validated_new_array_response_length": <number here>
}
```
The array length MUST be the same from the beginning, not larger, not smaller!
"""
GEMINI_EXTENDED_PROMPT = """And here, you will also be given further context to refine your corrections, in the form of the description of the audio. The description could possibly contain valuable information, such as clear subject names that would'nt have been clear, just from the transcriptions alone. Here is the description of the audio :
```
{}
```"""
# TODO: add ability to fetch from the video's description, and add its description for further context to the model.

# if start >= speech_start && end <= speech_end || start < speech_start && end >= speech_start || start <= speech_end && end > speech_end || start <= speech_start && end >= speech_end
def collect_speech_in_range(start_speech: float, end_speech: float, words: list[dict[str, any]]) -> list[dict[str, any]] :
    in_range = []
    for w in words:
        # works first try; nothing I make works first try :)
        if (w["start"] >= start_speech and w["end"] <= end_speech) or \
            (w["start"] < start_speech and w["end"] >= start_speech) or \
            (w["start"] <= end_speech and w["end"] > end_speech) or \
            (w["start"] <= start_speech and w["end"] >= end_speech):
            in_range.append(w)
    return in_range

def write_srt(segments: list[dict[str, any]], path: str) -> None:
    subs = []
    for k, v in enumerate(segments):
        subs.append(srt.Subtitle(
            index=k+1,
            start=datetime.timedelta(
                seconds=v["start"]
            ),
            end=datetime.timedelta(
                seconds=v["end"]
            ),
            content=v["words"]
        ))
    with open(path, "w") as fh:
        fh.write(srt.compose(subs))

def silerovad_timestamps(path: str, thresh: float) -> list[dict[str, any]]:
    vad = silero_vad.load_silero_vad()
    speech_timestamps = silero_vad.get_speech_timestamps(
        audio=silero_vad.read_audio(path),
        model=vad,
        return_seconds=True,
        time_resolution=3,
        threshold=thresh
    )
    return speech_timestamps

def captions_segment(timestamps: list[dict[str, any]], words: list[dict[str, any]], caption_preferred_max_length: int) -> list[dict[str, any]]:
    segmented = []
    for k, v in enumerate(timestamps):
        in_range = collect_speech_in_range(
            start_speech=v["start"],
            end_speech=v["end"],
            words=words
        )
        # print(f"{v["start"]} - {v["end"]} : {in_range}")

        # if this gets too long still...
        # max length target of 128 chars
        # loop every element, contain a comma or a period
        # " ".join(v.get("word") for v in in_range[last:k])
        # v["end"]
        last = 0
        for i, ire in enumerate(in_range):
            # https://stackoverflow.com/a/6531704
            if any(char in ire["word"] for char in [".", ","]) or i + 1 == len(in_range):
                chk = " ".join(v.get("word") for v in in_range[last:i+1])
                if len(chk) > caption_preferred_max_length or i + 1 == len(in_range):
                    segmented.append({
                        "start": in_range[last]["start"],
                        "end": in_range[i]["end"],
                        "words": chk
                    })
                    last = i
    return segmented

# fix desync between Whisper and SileroVAD
def captions_segment_fix(segments: list[dict[str, any]]) -> list[dict[str, any]]:
    seg_cpy = segments.copy()
    for k in range(len(seg_cpy)):
        if k + 1 == len(seg_cpy):
            continue

        if seg_cpy[k]["end"] > seg_cpy[k+1]["start"]:
            seg_cpy[k+1]["start"] = seg_cpy[k]["end"]
            # typecast for annotation
            seg_cpy[k+1]["words"] = " ".join(str(seg_cpy[k+1]["words"]).split(" ")[1:])
    return seg_cpy


# If some transcriptions are missing, its either SileroVAD not capturing speech segments, "dropping" them

def main():
    parser = argparse.ArgumentParser(
        description="A program to generate captions for audio files"
    )
    parser.add_argument("-q", "--groq-key", required=True, type=str, help="The Groq API key for calling Whisper")
    parser.add_argument("-g", "--gemini-key", required=True, type=str, help="The Gemini API key for calling the caption-correcting model")
    parser.add_argument("input", type=str, help="The input audio file to be captioned")
    parser.add_argument("-o", "--output", type=str, default="out.srt", help="The output caption file (SRT)")
    parser.add_argument("--gemini-model", type=str, default="gemma-3-27b-it", help="The Gemini model to be used for caption-correction")
    parser.add_argument("--gemini-temp", type=float, default=0.0, help="The temperature to use for the Gemini model")
    parser.add_argument("--groq-model", type=str, default="whisper-large-v3-turbo", help="The Groq Whisper model to be used for audio transcription")
    parser.add_argument("--silero-threshold", type=float, default=0.4, help="Threshold to use for SileroVAD timestamp processing")
    parser.add_argument("--segment-preferred-length", type=int, default=96, help="The preferred segment character length. If the transcription result returns longer than this, the program will try to split it off into multiple segments")
    parser.add_argument("--audio-description", type=str, help="The audio description to help the caption-correcting model to have more context over the audio")

    args = parser.parse_args()

    groq_key: str = args.groq_key
    gemini_key: str = args.gemini_key
    audio_input: str = args.input
    caption_out: str = args.output
    gemini_model: str = args.gemini_model
    gemini_temp: float = args.gemini_temp
    groq_model: str = args.groq_model
    silero_threshold: float = args.silero_threshold
    caption_preferred_max_length: int = args.segment_preferred_length
    audio_desc: str | None = args.audio_description

    print("loading and timestamping with SileroVAD...")
    start_silerovad = time.time()
    speech_timestamps = silerovad_timestamps(audio_input, silero_threshold)
    print(f"took {(time.time() - start_silerovad):.3f} seconds to load and timestamp with SileroVAD")

    groq_client = groq.Groq(
        api_key=groq_key
    )
    with open(audio_input, "rb") as fh:
        print("start transcription with Whisper (Groq API)...")
        start_groq = time.time()
        trans_res = groq_client.audio.transcriptions.create(
            file=(audio_input, fh.read()),
            model=groq_model,
            temperature=0.0,
            response_format="verbose_json",
            language="en",
            timestamp_granularities=["word"]
        )
    print(f"took {(time.time() - start_groq):.3f} seconds to transcribe with Whisper (Groq API)")

    whisper_word_data: dict[str, any] = trans_res.model_dump(include={"words"})["words"]

    corrected_segments = captions_segment_fix(captions_segment(
        timestamps=speech_timestamps,
        words=whisper_word_data,
        caption_preferred_max_length=caption_preferred_max_length
    ))

    google_client = genai.Client(
        api_key=gemini_key
    )
    # join abuse
    gemini_prompt = "\n\n".join([GEMINI_BASE_PROMPT, GEMINI_EXTENDED_PROMPT.format(audio_desc)] if audio_desc != None else [GEMINI_BASE_PROMPT])
    gemini_obj = {
        "list": [w["words"] for w in corrected_segments]
    }

    print("start correction with Gemini via API...")
    start_gemini = time.time()
    gemini_response = google_client.models.generate_content(
        model=gemini_model,
        contents="\n\n".join([gemini_prompt, json.dumps(gemini_obj)]),
        config=genai.types.GenerateContentConfig(
            temperature=gemini_temp
        )
    )
    print(f"took {(time.time() - start_gemini):.3f} seconds to correct with Gemini via API")

    # remove Markdown syntax and stuff
    # HACK: replace this char "’" to this "'", because Gemma's response to the apostrophe is different
    fixed_obj = json.loads(gemini_response.text.removeprefix("```json").removesuffix("```").replace("\u2019", "'"))["response"]

    # TODO: add handling in case the length returned by Gemma isn't the same

    final = corrected_segments.copy()
    if len(final) != len(fixed_obj):
        # HACK: really should do more! it's 2 AM here.
        print(f"[WARNINGWARNINGWARNING] final's length is {len(final)}, while fixed_obj's length is {len(fixed_obj)}, resorting to padding!")
        final += [""] * (max(len(final), len(fixed_obj)) - len(final))
        fixed_obj += [""] * (max(len(final), len(fixed_obj)) - len(fixed_obj))

    for i in range(len(final)):
        final[i]["words"] = fixed_obj[i]

    write_srt(final, caption_out)

if __name__ == "__main__":
    main()