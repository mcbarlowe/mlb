# MLB Data Transformation Classes

This directory contains data transformation classes that convert MLB API JSON responses into structured tabular formats (DataFrames) suitable for analysis and storage.

## Available Classes

### GameFeedData
Transforms GUMBO (Game Feed) JSON responses into pitch-level DataFrames.

**Features:**
- Flattens nested play-by-play data
- Extracts pitch-level metrics (velocity, spin rate, break, coordinates)
- Includes at-bat context (batter, pitcher, score, runners on base)
- Supports saving to Parquet, CSV, or JSON formats

**Output Schema:** 70+ columns including:
- Play context: event, inning, half_inning, outs, score
- Matchup: batter_id, pitcher_id, bat_side, throw_side
- Pitch data: pitch_type, speed, spin_rate, zone
- PITCHf/x: coordinates, velocities, accelerations, break metrics

**Usage:**
```python
from mlb.data import GameFeedData

transformer = GameFeedData()
pitches_df = transformer.transform(game_json, game_pk=716828)
transformer.save(pitches_df, Path("output.parquet"))
```

---

### LinescoreData
Transforms linescore data into inning-by-inning DataFrames.

**Features:**
- Separate rows for each team per inning
- Includes runs, hits, errors, left on base
- Captures game state metadata

**Output Schema:**
- game_pk, inning, inning_ordinal, team_type
- runs, hits, errors, left_on_base
- current_inning, inning_state, inning_half

**Usage:**
```python
from mlb.data import LinescoreData

transformer = LinescoreData()
linescore_df = transformer.transform(game_json, game_pk=716828)
transformer.save(linescore_df, Path("linescore.csv"), format="csv")
```

---

### BoxscoreData
Transforms boxscore data into player-level statistics DataFrames.

**Features:**
- Separate transformations for batting, pitching, fielding
- `transform_all()` method for comprehensive extraction
- Includes both game and season statistics

**Output Schema (per stat type):**
- Player info: player_id, player_name, jersey_number, position
- Batting: AB, H, R, RBI, HR, BB, K, AVG, OBP, SLG, etc.
- Pitching: IP, H, R, ER, BB, K, HR, ERA, WHIP, etc.
- Fielding: assists, putouts, errors, chances

**Usage:**
```python
from mlb.data import BoxscoreData

transformer = BoxscoreData()

# Transform specific stat type for one team
batting_df = transformer.transform_batting(game_json, game_pk=716828, team_type="away")

# Transform all stats for both teams
all_stats = transformer.transform_all(game_json, game_pk=716828)
# Returns: {'away_batting': df, 'away_pitching': df, ...}
```

---

### ReferenceData
Transforms simple reference endpoint responses into DataFrames.

**Features:**
- Generic transformer for any reference endpoint
- Auto-detects array data in responses
- Flattens nested structures using `pd.json_normalize()`

**Supported Endpoints:**
- Positions, PitchTypes, EventTypes, GameTypes
- GameStatus, Venues, Sky, WindDirection
- Any other simple list-based endpoints

**Usage:**
```python
from mlb.data import ReferenceData
from mlb.endpoints.positions import Positions

positions_api = Positions()
positions_json = positions_api.get()

transformer = ReferenceData()
positions_df = transformer.transform(positions_json)
transformer.save(positions_df, Path("positions.parquet"))
```

---

## File Formats

All data classes support three output formats via the `save()` method:

### Parquet (Recommended)
- **Pros:** Efficient compression, preserves data types, fast I/O
- **Use for:** Large datasets, analytics, data lakes
- **Extension:** `.parquet`

### CSV
- **Pros:** Human-readable, widely supported
- **Use for:** Sharing data, manual inspection
- **Extension:** `.csv`

### JSON Lines
- **Pros:** Structured, streaming-friendly
- **Use for:** APIs, line-by-line processing
- **Extension:** `.json`

```python
# Choose format when saving
transformer.save(df, path, format="parquet")  # Default
transformer.save(df, path, format="csv")
transformer.save(df, path, format="json")
```

---

## Design Pattern

All transformation classes follow this pattern:

1. **Transform:** Convert JSON → DataFrame
2. **Save:** Write DataFrame to file

```python
# Standard workflow
transformer = SomeDataClass()
df = transformer.transform(api_json, **metadata)
transformer.save(df, output_path, format="parquet")
```

---

## Examples

See `examples/data_transformation_examples.py` for complete working examples of:
- Single game transformations
- Batch processing multiple games
- Reference data extraction
- Different output formats

---

## Integration with API Wrappers

These data classes are designed to work seamlessly with the API wrapper classes in `src/endpoints/`:

```python
from mlb.endpoints.game_feed import GameFeed
from mlb.data import GameFeedData

# Fetch data
api = GameFeed()
game_json = api.get(game_pk=716828)

# Transform data
transformer = GameFeedData()
df = transformer.transform(game_json, game_pk=716828)

# Save data
transformer.save(df, Path("output.parquet"))
```
