import requests
import collections
import re
import matplotlib.pyplot as plt

def scrape_orcid_profile(orcid_id):
    # 1. Setup the Public API URL (Full Record)
    # Using the root endpoint gets Person + Activities (Works/Employment)
    url = f"https://pub.orcid.org/v3.0/{orcid_id}"
    
    headers = {
        "Accept": "application/json",
        "User-Agent": "ORCID_Full_Scraper/1.0"
    }

    print(f"Fetching full profile for ORCID: {orcid_id}...")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return

    # --- PART A: Extract Personal Info ---
    person = data.get('person', {})
    activities = data.get('activities-summary', {})

    # Name
    try:
        name_data = person.get('name', {})
        given = name_data.get('given-names', {}).get('value', '')
        family = name_data.get('family-name', {}).get('value', '')
        full_name = f"{given} {family}".strip()
    except:
        full_name = "Unknown Name"

    # Affiliations (Employment)
    affiliations = []
    employments = activities.get('employments', {}).get('affiliation-group', [])
    for group in employments:
        # Get the summary of the first employment record in the group
        summaries = group.get('summaries', [])
        if summaries:
            summary = summaries[0].get('employment-summary', {})
            org_name = summary.get('organization', {}).get('name')
            
            # Get dates to see if it's current
            start_date = summary.get('start-date', {})
            start_year = start_date.get('year', {}).get('value') if start_date else "?"
            end_date = summary.get('end-date')
            end_year = "Present" if end_date is None else end_date.get('year', {}).get('value')
            
            affiliations.append(f"{org_name} ({start_year}-{end_year})")

    # --- PART B: Extract Works & Themes ---
    works_groups = activities.get('works', {}).get('group', [])
    works_by_year = collections.defaultdict(list)
    
    for group in works_groups:
        work_summary = group['work-summary'][0]
        title = work_summary.get('title', {}).get('title', {}).get('value', "")
        
        pub_date = work_summary.get('publication-date')
        if pub_date and pub_date.get('year'):
            year = pub_date.get('year').get('value')
        else:
            year = "Unknown"
            
        if title:
            works_by_year[year].append(title)

    # Theme Helper (Simple Keyword Frequency)
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 
                      'with', 'by', 'from', 'using', 'based', 'study', 'analysis', 'review',
                      'approach', 'method', 'system', 'model', 'data', 'via', 'application'])

    def get_themes(titles_list):
        words = []
        for t in titles_list:
            clean_text = re.sub(r'[^\w\s]', '', t.lower())
            for word in clean_text.split():
                if word not in stop_words and len(word) > 2:
                    words.append(word.capitalize())
        return ", ".join([w for w, c in collections.Counter(words).most_common(3)])

    # --- PART C: Output & Visualization ---
    
    # 1. Text Report
    print("\n" + "="*60)
    print(f"RESEARCHER: {full_name}")
    print(f"ORCID:      {orcid_id}")
    print("-" * 60)
    print("AFFILIATIONS:")
    for aff in affiliations:
        print(f" - {aff}")
    
    print("\nPUBLICATIONS SUMMARY:")
    print(f"{'Year':<10} | {'Count':<5} | {'Top Themes/Keywords'}")
    print("-" * 60)

    # Sort years for printing
    sorted_years = sorted([y for y in works_by_year.keys() if y != "Unknown"], reverse=True)
    if "Unknown" in works_by_year: sorted_years.append("Unknown")

    total_works = 0
    years_for_plot = []
    counts_for_plot = []

    for year in sorted_years:
        count = len(works_by_year[year])
        total_works += count
        themes = get_themes(works_by_year[year])
        print(f"{year:<10} | {count:<5} | {themes}")
        
        # Prepare data for plot (exclude Unknown for the chart)
        if year != "Unknown":
            years_for_plot.append(year)
            counts_for_plot.append(count)

    print("-" * 60)
    print(f"Total Works Found: {total_works}")
    print("="*60)

    # 2. Bar Plot
    if years_for_plot:
        # Sort chronologically for the chart
        years_sorted = years_for_plot[::-1] 
        counts_sorted = counts_for_plot[::-1]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(years_sorted, counts_sorted, color='skyblue', edgecolor='navy')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')

        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        plt.title(f'Publication History: {full_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        print("\nDisplaying plot...")
        plt.show()
    else:
        print("\nNot enough data to generate a plot.")

# --- Usage ---
# User provided ORCID: 0000-0002-8627-5974
user_orcid = input("Enter ORCID iD (e.g., 0000-0002-8627-5974): ").strip()
scrape_orcid_profile(user_orcid)