#!/bin/bash

set -eo pipefail

if [[ -z "$1" ]]; then
    echo "directory not specified, exiting"
    exit 1
fi
if [[ -z "$2" ]]; then
    echo "Groq API key not specified, exiting"
    exit 1
fi
if [[ -z "$3" ]]; then
    echo "Gemini API key not specified, exiting"
    exit 1
fi

if [[ ! -d "$1" ]]; then
    echo "directory doesn't exist?"
    exit 1
fi

# https://askubuntu.com/a/893927
SCRIPT_DIR=$(dirname "$(realpath "$0")")
DEBUG=0
if [[ ! -z "$4" ]]; then
    case "$4" in
        "--debug-noprocessing")
            DEBUG=1
            ;;
        "--debug-embedsrt")
            DEBUG=2
            ;;
        *)
            echo "unrecognized extra argument, exiting"
            exit 1
            ;;
    esac
fi

# use venv for the packages
source $SCRIPT_DIR/.venv/bin/activate

for vid in "$1/"*.mkv; do
    basedir="$(dirname "$vid")"
    if [[ ! -d "$basedir/out/" ]]; then
        mkdir "$basedir/out/"
    fi

    sub_check="$(ffprobe -hide_banner -i "$vid" -v quiet -select_streams s -show_entries stream=codec_name -print_format csv)"
    # https://stackoverflow.com/a/229606
    # skip videos that already has captions (only SRT for now)
    if [[ $sub_check == *"subrip"* ]]; then
        continue
    fi

    tmp_aud="$(mktemp --suffix=".opus")"
    tmp_srt="$(mktemp --suffix=".srt")"
    tmp_thumb="$(mktemp --suffix=".png")"

    audio_desc="$(ffprobe -hide_banner -i "$vid" -v quiet -select_streams s -show_entries format_tags=DESCRIPTION -of csv=p=0)"

    if [[ $DEBUG == 0 ]]; then
        ffmpeg -hide_banner -y -i "$vid" -map 0:a -c:a libopus -b:a 48k "$tmp_aud"
        python "$SCRIPT_DIR/trans.py" -q "$2" -g "$3" -o "$tmp_srt" --groq-model "whisper-large-v3" --audio-description "$audio_desc" "$tmp_aud"
    else
        echo "skipped processing $vid due to debug mode"
    fi

    if [[ $DEBUG == 2 ]]; then
        # https://stackoverflow.com/a/7875614
        cat > "$tmp_srt" <<EOF
1
00:00:01,000 --> 00:00:05,000
Test 1

2
00:00:09,500 --> 00:00:12,100
Test 2

3
00:00:15,000 --> 00:00:18,000
Test 3

4
00:00:21,000 --> 00:00:25,000
Test 4

5
00:00:28,000 --> 00:00:37,250
Test 5. Congratulations, we have embedded an English captions track, and set it as the default captions track, too.
EOF
    fi

    if [[ $DEBUG == 0 || $DEBUG == 2 ]]; then
        ffmpeg -hide_banner -y -i "$vid" -map 0:2 -c:v copy -update 1 -frames:v 1 "$tmp_thumb"
        ffmpeg -hide_banner -y -i "$vid" -i "$tmp_srt" -map 0:0 -map 0:1 -c:v copy -c:a copy -map 1 -c:s:0 srt -metadata:s:0 language="eng" -disposition:s:0 default -attach "$tmp_thumb" -metadata:s:t "mimetype=image/png" -metadata:s:t "filename=cover.png" "$basedir/out/$(basename "$vid")"
    fi

    rm "$tmp_thumb"
    rm "$tmp_srt"
    rm "$tmp_aud"
done