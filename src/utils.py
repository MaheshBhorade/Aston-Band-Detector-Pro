import csv
import datetime

def seconds_to_hms(sec):
    sec = int(sec)
    return f"{sec//3600:02}:{(sec%3600)//60:02}:{sec%60:02}"

def merge_segments(segments, merge_gap=15):
    """
    Merges segments of the same ad if they are close together.
    Following the logic from your production system.
    """
    if not segments:
        return []

    # Sort by start time
    segments.sort(key=lambda x: x[1])

    merged = [list(segments[0])]

    for seg in segments[1:]:
        ad_id, start, end, conf = seg
        last_ad, last_start, last_end, last_conf = merged[-1]

        if ad_id == last_ad and (start - last_end) <= merge_gap:
            merged[-1][2] = end
            merged[-1][3] = max(last_conf, conf)
        else:
            merged.append(list(seg))

    return [tuple(m) for m in merged]

def export_to_csv(results, output_path, channel_info=None):
    """
    Exports detections to CSV using the exact format from test_ad_detection.
    """
    channel_config = channel_info if channel_info else {}
    channel_id = channel_config.get('id', '1011')
    channel_name = channel_config.get('name', 'Shant TV')
    formatted_date = datetime.datetime.now().strftime("%Y-%m-%d")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "srno","masterfp","referencefp","date","time","channelid",
            "channel","starttime","endtime","duration",
            "confidence","labelid","labeltype","AdMasterID","programid"
        ])

        for sr, res in enumerate(results, 1):
            ad_id, start, end, conf = res
            duration = end - start
            
            writer.writerow([
                sr, "", "",
                formatted_date,
                seconds_to_hms(start),
                channel_id,
                channel_name,
                seconds_to_hms(start),
                seconds_to_hms(end),
                round(duration, 2),
                round(conf, 2),
                2, # labelid (Aston usually 2 in your system)
                "Secondary Element",
                ad_id,
                ""
            ])
    print(f"Report saved to {output_path}")
