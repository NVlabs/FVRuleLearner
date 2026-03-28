import re

def parse_textbook(text):
    lines = text.split('\n')
    data = {}
    current_chapter = None
    current_section = None
    current_rules = []

    chapter_pattern = re.compile(r'^Chapter\s+(\d+):\s+(.*)$')
    section_pattern = re.compile(r'^(\d+\.\d+)\s+(.*)$')
    rule_pattern = re.compile(r'^\s*•\s*Rule:\s*(.*)$')

    for line in lines:
        line = line.strip()

        # Check if this line indicates a chapter start
        ch_match = chapter_pattern.match(line)
        if ch_match:
            # If we were tracking a previous section and rules, save them before moving on
            if current_section and current_chapter:
                if current_section not in data[current_chapter]:
                    data[current_chapter][current_section] = []
                data[current_chapter][current_section].extend(current_rules)

            # Start a new chapter
            chapter_num = ch_match.group(1)
            chapter_title = ch_match.group(2)
            current_chapter = f"Chapter {chapter_num}: {chapter_title}"
            data[current_chapter] = {}
            current_section = None
            current_rules = []
            continue

        # Check if this line indicates a new subsection
        sec_match = section_pattern.match(line)
        if sec_match and current_chapter:
            # If we were tracking a previous section and rules, store them
            if current_section:
                data[current_chapter][current_section] = current_rules

            # Start a new subsection
            section_num = sec_match.group(1)
            section_title = sec_match.group(2)
            current_section = f"{section_num} {section_title}"
            current_rules = []
            continue

        # Check if this line indicates a rule
        rule_match = rule_pattern.match(line)
        if rule_match and current_chapter and current_section:
            rule_text = "Rule: " + rule_match.group(1).strip()
            current_rules.append(rule_text)
            continue

        # Other lines (introductions, examples, etc.) are not rules and are not stored.

    # At the end, if there is a last section being tracked, store its rules
    if current_section and current_chapter:
        data[current_chapter][current_section] = current_rules

    return data


def get_chapters(data):
    return list(data.keys())


def get_sections(data, chapter_name):
    if chapter_name in data:
        return list(data[chapter_name].keys())
    return []


def get_rules(data, chapter_name, section_name=None):
    # Return rules from a specific chapter and optionally a specific section
    if chapter_name not in data:
        return []

    if section_name is None:
        # Return all rules from all sections of the chapter
        all_rules = []
        for sec in data[chapter_name]:
            all_rules.extend(data[chapter_name][sec])
        return all_rules
    else:
        # Return rules from the specified section only
        return data[chapter_name].get(section_name, [])


def is_chapter_related(chapter_name, query):
    # Simple heuristic: check if query words appear in chapter_name
    query_lower = query.lower()
    chapter_lower = chapter_name.lower()
    return any(word in chapter_lower for word in query_lower.split())


def retrieve_rules_based_on_query(data, query):
    # 1. Find related chapters
    related_chapters = [ch for ch in data if is_chapter_related(ch, query)]
    if not related_chapters:
        # If no chapter is related, return empty
        return {}

    # For simplicity, return rules from the first related chapter
    chapter = related_chapters[0]
    return {chapter: get_rules(data, chapter)}


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Read the textbook content from the given file
    file_path = "/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/suggestions_pkl/textbook.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        textbook_text = f.read()

    # Parse the text
    data = parse_textbook(textbook_text)

    # Print all chapters
    chapters = get_chapters(data)
    print("Chapters found:")
    for ch in chapters:
        print(" ", ch)

    # Let's say we want to find rules in Chapter 1
    chap1 = "Chapter 1: Fundamental Principles of SystemVerilog Assertions"
    if chap1 in data:
        sections_chap1 = get_sections(data, chap1)
        print("\nSections in Chapter 1:")
        for sec in sections_chap1:
            print(" ", sec)

        # Get rules from Section 1.1
        sec11 = "1.1 Direct Translation"
        rules_sec11 = get_rules(data, chap1, sec11)
        print("\nRules in Chapter 1, Section 1.1:")
        for r in rules_sec11:
            print(" ", r)

    # Example: Retrieve rules based on a query
    query = "Fundamental Principles"
    related_rules = retrieve_rules_based_on_query(data, query)
    print(f"\nRelated rules for query '{query}':")
    for ch_name, rules_list in related_rules.items():
        print(ch_name)
        for r in rules_list:
            print(" ", r)
