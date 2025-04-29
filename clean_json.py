import json
import os

input_dir = './tweets_json'
output_dir = './cleaned_tweets_json'
log_dir = './cleaning_logs'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def is_valid_json(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False
    
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        log_filename = filename.replace('/', '_').replace('\\', '_') + '.log'
        log_path = os.path.join(log_dir, log_filename)

        log_lines = []

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line_number, line in enumerate(infile, 1):
                original_line = line.strip()
                if not original_line:
                    continue
                if original_line.startswith('{') and is_valid_json(original_line):
                    outfile.write(original_line + '\n')
                else:
                    log_lines.append(f"[Line {line_number}] Skipped: {original_line[:100]}")

        # Write log if there were errors, otherwise skip
        if log_lines:
            with open(log_path, 'w', encoding='utf-8') as log_file:
                log_file.write("\n".join(log_lines))
        else:
            if os.path.exists(log_path):
                os.remove(log_path)
                    