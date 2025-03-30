import json

# Load the extracted tables JSON
with open("extracted_tables.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Define the target JSON format
formatted_data = []

for entry in data:
    table = entry['table']
    if len(table) > 2:
        course_info = table[1]  # Course information row
        description = table[2][0].replace('\n', ' ').strip() if table[2][0] else ""
        
        # Create the formatted entry
        course_entry = {
            "course_code": course_info[0] if len(course_info) > 0 else None,
            "title": course_info[2] if len(course_info) > 2 else None,
            "instructor": course_info[3].replace('\n', '、').strip() if len(course_info) > 3 else None,
            "faculty": course_info[4] if len(course_info) > 4 else None,
            "day_period": course_info[5] if len(course_info) > 5 else None,
            "credits": int(course_info[6]) if len(course_info) > 6 and course_info[6].isdigit() else None,
            "target": course_info[7].replace('\n', ' ') if len(course_info) > 7 else None,
            "description": description,
            "evaluation": "",
            "textbook": "",
            "guidance": ""
        }
        
        # Extract additional information from description if available
        if "成績評価方法" in description:
            eval_split = description.split("成績評価方法")
            course_entry["description"] = eval_split[0].strip()
            if len(eval_split) > 1:
                eval_parts = eval_split[1].split("教科書")
                course_entry["evaluation"] = eval_parts[0].strip()
                if len(eval_parts) > 1:
                    text_parts = eval_parts[1].split("ガイダンス")
                    course_entry["textbook"] = text_parts[0].strip()
                    if len(text_parts) > 1:
                        course_entry["guidance"] = text_parts[1].strip()

        formatted_data.append(course_entry)

# Save the formatted JSON
output_path = "formatted_courses.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print(f"データが {output_path} に保存されました！")
