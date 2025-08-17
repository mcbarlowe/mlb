import json

from src.endpoints.schedule import Schedule

schedule = Schedule()

for season in range(2018, 2026):
    print(f"Fetching schedule for season {season}...")
    try:
        data = schedule.get(sportId=1,season=season)
        with open(f"schedule_{season}.json", "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error fetching schedule for season {season}: {e}")
