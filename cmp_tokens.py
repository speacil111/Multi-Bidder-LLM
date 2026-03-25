import re
import csv

_EMOJI_RE = re.compile(
    "[\U00010000-\U0010FFFF"
    "\uFE00-\uFE0F"
    "\u200D"
    "\u2600-\u27BF"
    "\u2B50-\u2B55"
    "\u231A-\u231B"
    "\u23CF-\u23F3"
    "\u2934-\u2935"
    "\u25AA-\u25FE"
    "\u3030\u303D"
    "]+"
)

def remove_emoji(text):
    cleaned = _EMOJI_RE.sub('', text)
    return re.sub(r' {2,}', ' ', cleaned).strip()


def count_words(lines):
    merged = " ".join(lines)
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", merged))


def process_report(input_file, csv_file, txt_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    runs = content.split('------------------------------------------------------------')
    
    csv_data = [["Run", "Hilton_Count", "Delta_Count", "Hilton_Words", "Delta_Words"]]
    txt_data = []

    for run in runs:
        if "Run " not in run:
            continue
            
        run_match = re.search(r'(Run \d+)', run)
        if not run_match:
            continue
        run_name = run_match.group(1)
        
        text_content = re.sub(r'\\', '', run)
        text_content = remove_emoji(text_content)
        
        paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
        
        delta_ads = []
        hilton_ads = []
        
        for p in paragraphs:
            has_delta = "Delta" in p
            has_hilton = "Hilton" in p
            
            if has_delta and not has_hilton:
                delta_ads.append(p)
            elif has_hilton and not has_delta:
                hilton_ads.append(p)
            elif has_delta and has_hilton:
                sentences = p.split('. ')
                for sentence in sentences:
                    if "Delta" in sentence and "Hilton" in sentence:
                        delta_part = re.sub(r'and stay at Hilton[^—\-]+', '', sentence, flags=re.IGNORECASE)
                        delta_ads.append(delta_part.strip())
                        
                        hilton_part = re.sub(r'Book your flight with Delta[^a-z]+and ', '', sentence, flags=re.IGNORECASE)
                        hilton_ads.append(hilton_part.strip())
                    else:
                        if "Delta" in sentence:
                            delta_ads.append(sentence)
                        if "Hilton" in sentence:
                            hilton_ads.append(sentence)

        hit_match = re.search(r'hit_hilton=(\d+),\s*hit_delta=(\d+)', run)
        if hit_match:
            hilton_count = int(hit_match.group(1))
            delta_count = int(hit_match.group(2))
        else:
            hilton_count = len(hilton_ads)
            delta_count = len(delta_ads)
            
        hilton_words = count_words(hilton_ads)
        delta_words = count_words(delta_ads)

        csv_data.append([run_name, hilton_count, delta_count, hilton_words, delta_words])
        
        if hilton_count > 0 or delta_count > 0:
            txt_data.append(f"=== {run_name} ===")
            if delta_ads:
                txt_data.append(f"[Delta 广告词] ({delta_words} words)")
                txt_data.extend(delta_ads)
            if hilton_ads:
                txt_data.append(f"[Hilton 广告词] ({hilton_words} words)")
                txt_data.extend(hilton_ads)
            txt_data.append("\n")

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_data))

process_report('./double_500_[1-4]_unified_output/report.txt', 'ad_counts.csv', 'ad_details.txt')
