
CREATE TABLE IF NOT EXISTS backfill_game_progress (
    source_key character varying NOT NULL,
    game_pk integer NOT NULL,
    season integer NOT NULL,
    status character varying NOT NULL,
    last_error character varying,
    loaded_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE IF NOT EXISTS batting (
    game_pk integer NOT NULL,
    team_type character varying NOT NULL,
    player_id integer NOT NULL,
    player_name character varying,
    jersey_number character varying,
    position_code character varying,
    position_name character varying,
    position_abbrev character varying,
    batting_order character varying,
    summary character varying,
    gamesplayed integer,
    flyouts integer,
    groundouts integer,
    airouts integer,
    runs integer,
    doubles integer,
    triples integer,
    homeruns integer,
    strikeouts integer,
    baseonballs integer,
    intentionalwalks integer,
    hits integer,
    hitbypitch integer,
    atbats integer,
    caughtstealing integer,
    stolenbases integer,
    stolenbasepercentage numeric,
    groundintodoubleplay integer,
    groundintotripleplay integer,
    plateappearances integer,
    totalbases integer,
    rbi integer,
    leftonbase integer,
    sacbunts integer,
    sacflies integer,
    catchersinterference integer,
    pickoffs integer,
    atbatsperhomerun numeric,
    popouts integer,
    lineouts integer,
    note character varying
);

CREATE TABLE IF NOT EXISTS bulk_backfill_progress (
    season integer NOT NULL,
    total_games integer NOT NULL,
    status character varying NOT NULL,
    last_error character varying,
    loaded_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE IF NOT EXISTS event_types (
    code character varying NOT NULL,
    description character varying
);

CREATE TABLE IF NOT EXISTS fielding (
    game_pk integer NOT NULL,
    team_type character varying NOT NULL,
    player_id integer NOT NULL,
    player_name character varying,
    jersey_number character varying,
    position_code character varying NOT NULL,
    position_name character varying,
    position_abbrev character varying,
    gamesstarted integer,
    caughtstealing integer,
    stolenbases integer,
    stolenbasepercentage numeric,
    caughtstealingpercentage numeric,
    assists integer,
    putouts integer,
    errors integer,
    chances integer,
    fielding numeric,
    passedball integer,
    pickoffs integer
);

CREATE TABLE IF NOT EXISTS game_types (
    id character varying NOT NULL,
    description character varying
);

CREATE TABLE IF NOT EXISTS games (
    game_pk integer NOT NULL,
    game_id character varying,
    season integer,
    season_display character varying,
    game_type character varying,
    gameday_type character varying,
    game_number integer,
    double_header character varying,
    tiebreaker character varying,
    calendar_event_id character varying,
    game_date date,
    original_date date,
    game_datetime timestamp with time zone,
    game_time character varying,
    ampm character varying,
    day_night character varying,
    abstract_game_state character varying,
    coded_game_state character varying,
    detailed_state character varying,
    status_code character varying,
    start_time_tbd boolean,
    abstract_game_code character varying,
    venue_id integer,
    weather_condition character varying,
    weather_temp character varying,
    weather_wind character varying,
    attendance integer,
    first_pitch timestamp with time zone,
    game_duration_minutes integer,
    away_team_id integer,
    away_team_wins integer,
    away_team_losses integer,
    away_team_winning_percentage numeric,
    away_team_division_leader boolean,
    away_team_games_played integer,
    home_team_id integer,
    home_team_wins integer,
    home_team_losses integer,
    home_team_winning_percentage numeric,
    home_team_division_leader boolean,
    home_team_games_played integer,
    away_probable_pitcher_id integer,
    away_probable_pitcher_name character varying,
    home_probable_pitcher_id integer,
    home_probable_pitcher_name character varying,
    has_challenges boolean,
    away_reviews_remaining integer,
    away_reviews_used integer,
    home_reviews_remaining integer,
    home_reviews_used integer,
    no_hitter boolean,
    perfect_game boolean,
    away_team_no_hitter boolean,
    away_team_perfect_game boolean,
    home_team_no_hitter boolean,
    home_team_perfect_game boolean
);

CREATE TABLE IF NOT EXISTS linescore (
    game_pk integer NOT NULL,
    inning integer NOT NULL,
    inning_ordinal character varying,
    team_type character varying NOT NULL,
    runs integer,
    hits integer,
    errors integer,
    left_on_base integer,
    current_inning integer,
    inning_state character varying,
    inning_half character varying,
    scheduled_innings integer
);

CREATE TABLE IF NOT EXISTS pitch_types (
    code character varying NOT NULL,
    description character varying
);

CREATE TABLE IF NOT EXISTS pitches (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
)
PARTITION BY LIST (season);

CREATE TABLE IF NOT EXISTS pitches_p2009 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2010 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2011 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2012 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2013 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2014 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2015 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2016 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2017 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2018 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2019 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2020 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2021 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2022 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2023 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2024 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2025 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_p2026 (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitches_pdefault (
    game_pk integer NOT NULL,
    season integer NOT NULL,
    game_type character varying,
    double_header character varying,
    game_number integer,
    game_date timestamp with time zone,
    day_night character varying,
    away_team_id integer,
    away_team_name character varying,
    home_team_id integer,
    home_team_name character varying,
    venue_id integer,
    venue_name character varying,
    weather_condition character varying,
    weather_temp double precision,
    weather_wind character varying,
    event character varying,
    event_type character varying,
    description character varying,
    rbi integer,
    away_score integer,
    home_score integer,
    is_out boolean,
    at_bat_index integer NOT NULL,
    half_inning character varying,
    inning integer,
    batter_id integer,
    batter_name character varying,
    bat_side character varying,
    pitcher_id integer,
    pitcher_name character varying,
    throw_side character varying,
    pitch_number integer NOT NULL,
    pitch_call_description character varying,
    is_in_play boolean,
    is_strike boolean,
    is_ball boolean,
    pitch_type character varying,
    pitch_type_code character varying,
    count_after_pitch character varying,
    outs integer,
    play_id character varying,
    pitch_start_time timestamp with time zone,
    pitch_end_time timestamp with time zone,
    pitch_start_speed double precision,
    pitch_end_speed double precision,
    pitch_strike_zone_top double precision,
    pitch_strike_zone_bottom double precision,
    pitch_zone double precision,
    ay double precision,
    az double precision,
    pfxx double precision,
    pfxz double precision,
    px double precision,
    pz double precision,
    vx0 double precision,
    vy0 double precision,
    vz0 double precision,
    x double precision,
    y double precision,
    x0 double precision,
    y0 double precision,
    z0 double precision,
    ax double precision,
    break_angle double precision,
    break_length double precision,
    break_y double precision,
    break_vertical double precision,
    break_vertical_induced double precision,
    break_horizontal double precision,
    spin_rate double precision,
    spin_direction character varying,
    is_runner_on_first boolean,
    runner_on_first_id integer,
    is_runner_on_second boolean,
    runner_on_second_id integer,
    is_runner_on_third boolean,
    runner_on_third_id integer
);

CREATE TABLE IF NOT EXISTS pitching (
    game_pk integer NOT NULL,
    team_type character varying NOT NULL,
    player_id integer NOT NULL,
    player_name character varying,
    jersey_number character varying,
    position_code character varying,
    position_name character varying,
    position_abbrev character varying,
    summary character varying,
    gamesplayed integer,
    gamesstarted integer,
    flyouts integer,
    groundouts integer,
    airouts integer,
    runs integer,
    doubles integer,
    triples integer,
    homeruns integer,
    strikeouts integer,
    baseonballs integer,
    intentionalwalks integer,
    hits integer,
    hitbypitch integer,
    atbats integer,
    caughtstealing integer,
    stolenbases integer,
    stolenbasepercentage numeric,
    numberofpitches integer,
    inningspitched character varying,
    wins integer,
    losses integer,
    saves integer,
    saveopportunities integer,
    holds integer,
    blownsaves integer,
    earnedruns integer,
    battersfaced integer,
    outs integer,
    gamespitched integer,
    completegames integer,
    shutouts integer,
    pitchesthrown integer,
    balls integer,
    strikes integer,
    strikepercentage numeric,
    hitbatsmen integer,
    balks integer,
    wildpitches integer,
    pickoffs integer,
    rbi integer,
    gamesfinished integer,
    runsscoredper9 numeric,
    homerunsper9 numeric,
    inheritedrunners integer,
    inheritedrunnersscored integer,
    catchersinterference integer,
    sacbunts integer,
    sacflies integer,
    passedball integer,
    popouts integer,
    lineouts integer,
    note character varying
);

CREATE TABLE IF NOT EXISTS players (
    player_id integer NOT NULL,
    full_name character varying,
    first_name character varying,
    last_name character varying,
    middle_name character varying,
    use_name character varying,
    boxscore_name character varying,
    nick_name character varying,
    name_first_last character varying,
    name_slug character varying,
    first_last_name character varying,
    last_first_name character varying,
    last_init_name character varying,
    init_last_name character varying,
    full_fml_name character varying,
    full_lfm_name character varying,
    primary_number character varying,
    birth_date date,
    current_age integer,
    birth_city character varying,
    birth_state_province character varying,
    birth_country character varying,
    height character varying,
    weight integer,
    active boolean,
    primary_position_code character varying,
    primary_position_name character varying,
    primary_position_type character varying,
    primary_position_abbrev character varying,
    bat_side_code character varying,
    bat_side_description character varying,
    pitch_hand_code character varying,
    pitch_hand_description character varying,
    draft_year integer,
    mlb_debut_date date,
    strike_zone_top double precision,
    strike_zone_bottom double precision
);

CREATE TABLE IF NOT EXISTS positions (
    code character varying NOT NULL,
    name character varying,
    type character varying,
    abbreviation character varying
);

CREATE TABLE IF NOT EXISTS teams (
    team_id integer NOT NULL,
    team_name character varying,
    team_code character varying,
    file_code character varying,
    abbreviation character varying,
    team_name_short character varying,
    location_name character varying,
    first_year_of_play integer,
    league_id integer,
    league_name character varying,
    division_id integer,
    division_name character varying,
    sport_id integer,
    sport_name character varying,
    venue_id integer,
    venue_name character varying,
    spring_league_id integer,
    spring_league_name character varying,
    spring_league_abbrev character varying,
    parent_org_name character varying,
    parent_org_id integer,
    all_star_status boolean,
    active boolean
);

CREATE TABLE IF NOT EXISTS venues (
    venue_id integer NOT NULL,
    venue_name character varying,
    venue_link character varying,
    active boolean,
    season integer,
    address character varying,
    city character varying,
    state character varying,
    state_abbrev character varying,
    country character varying,
    postal_code character varying,
    latitude double precision,
    longitude double precision,
    elevation double precision,
    azimuth_angle double precision,
    timezone_id character varying,
    timezone character varying,
    timezone_offset double precision,
    capacity integer,
    turf_type character varying,
    roof_type character varying,
    left_line double precision,
    left_center double precision,
    center double precision,
    right_center double precision,
    right_line double precision
);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2009 FOR VALUES IN (2009);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2010 FOR VALUES IN (2010);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2011 FOR VALUES IN (2011);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2012 FOR VALUES IN (2012);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2013 FOR VALUES IN (2013);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2014 FOR VALUES IN (2014);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2015 FOR VALUES IN (2015);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2016 FOR VALUES IN (2016);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2017 FOR VALUES IN (2017);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2018 FOR VALUES IN (2018);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2019 FOR VALUES IN (2019);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2020 FOR VALUES IN (2020);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2021 FOR VALUES IN (2021);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2022 FOR VALUES IN (2022);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2023 FOR VALUES IN (2023);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2024 FOR VALUES IN (2024);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2025 FOR VALUES IN (2025);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_p2026 FOR VALUES IN (2026);

ALTER TABLE ONLY pitches ATTACH PARTITION pitches_pdefault DEFAULT;

ALTER TABLE ONLY backfill_game_progress
    ADD CONSTRAINT backfill_game_progress_pkey PRIMARY KEY (source_key);

ALTER TABLE ONLY batting
    ADD CONSTRAINT batting_pk PRIMARY KEY (game_pk, player_id, team_type);

ALTER TABLE ONLY bulk_backfill_progress
    ADD CONSTRAINT bulk_backfill_progress_pkey PRIMARY KEY (season);

ALTER TABLE ONLY event_types
    ADD CONSTRAINT event_types_pkey PRIMARY KEY (code);

ALTER TABLE ONLY fielding
    ADD CONSTRAINT fielding_pk PRIMARY KEY (game_pk, player_id, team_type, position_code);

ALTER TABLE ONLY game_types
    ADD CONSTRAINT game_types_pkey PRIMARY KEY (id);

ALTER TABLE ONLY games
    ADD CONSTRAINT games_pkey PRIMARY KEY (game_pk);

ALTER TABLE ONLY linescore
    ADD CONSTRAINT linescore_pk PRIMARY KEY (game_pk, inning, team_type);

ALTER TABLE ONLY pitch_types
    ADD CONSTRAINT pitch_types_pkey PRIMARY KEY (code);

ALTER TABLE ONLY pitches
    ADD CONSTRAINT pitches_pk PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2009
    ADD CONSTRAINT pitches_p2009_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2010
    ADD CONSTRAINT pitches_p2010_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2011
    ADD CONSTRAINT pitches_p2011_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2012
    ADD CONSTRAINT pitches_p2012_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2013
    ADD CONSTRAINT pitches_p2013_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2014
    ADD CONSTRAINT pitches_p2014_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2015
    ADD CONSTRAINT pitches_p2015_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2016
    ADD CONSTRAINT pitches_p2016_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2017
    ADD CONSTRAINT pitches_p2017_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2018
    ADD CONSTRAINT pitches_p2018_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2019
    ADD CONSTRAINT pitches_p2019_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2020
    ADD CONSTRAINT pitches_p2020_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2021
    ADD CONSTRAINT pitches_p2021_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2022
    ADD CONSTRAINT pitches_p2022_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2023
    ADD CONSTRAINT pitches_p2023_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2024
    ADD CONSTRAINT pitches_p2024_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2025
    ADD CONSTRAINT pitches_p2025_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_p2026
    ADD CONSTRAINT pitches_p2026_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitches_pdefault
    ADD CONSTRAINT pitches_pdefault_pkey PRIMARY KEY (season, game_pk, at_bat_index, pitch_number);

ALTER TABLE ONLY pitching
    ADD CONSTRAINT pitching_pk PRIMARY KEY (game_pk, player_id, team_type);

ALTER TABLE ONLY players
    ADD CONSTRAINT players_pkey PRIMARY KEY (player_id);

ALTER TABLE ONLY positions
    ADD CONSTRAINT positions_pkey PRIMARY KEY (code);

ALTER TABLE ONLY teams
    ADD CONSTRAINT teams_pkey PRIMARY KEY (team_id);

ALTER TABLE ONLY venues
    ADD CONSTRAINT venues_pkey PRIMARY KEY (venue_id);

CREATE INDEX IF NOT EXISTS idx_backfill_game_progress_status ON backfill_game_progress USING btree (status);

CREATE INDEX IF NOT EXISTS idx_batting_player ON batting USING btree (player_id);

CREATE INDEX IF NOT EXISTS idx_bulk_backfill_progress_status ON bulk_backfill_progress USING btree (status);

CREATE INDEX IF NOT EXISTS idx_fielding_player ON fielding USING btree (player_id);

CREATE INDEX IF NOT EXISTS idx_pitches_batter_season ON ONLY pitches USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS idx_pitches_pitcher_season ON ONLY pitches USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS idx_pitches_ptype_code ON ONLY pitches USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS idx_pitching_player ON pitching USING btree (player_id);

CREATE INDEX IF NOT EXISTS pitches_p2009_batter_id_season_idx ON pitches_p2009 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2009_pitch_type_code_idx ON pitches_p2009 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2009_pitcher_id_season_idx ON pitches_p2009 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2010_batter_id_season_idx ON pitches_p2010 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2010_pitch_type_code_idx ON pitches_p2010 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2010_pitcher_id_season_idx ON pitches_p2010 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2011_batter_id_season_idx ON pitches_p2011 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2011_pitch_type_code_idx ON pitches_p2011 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2011_pitcher_id_season_idx ON pitches_p2011 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2012_batter_id_season_idx ON pitches_p2012 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2012_pitch_type_code_idx ON pitches_p2012 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2012_pitcher_id_season_idx ON pitches_p2012 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2013_batter_id_season_idx ON pitches_p2013 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2013_pitch_type_code_idx ON pitches_p2013 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2013_pitcher_id_season_idx ON pitches_p2013 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2014_batter_id_season_idx ON pitches_p2014 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2014_pitch_type_code_idx ON pitches_p2014 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2014_pitcher_id_season_idx ON pitches_p2014 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2015_batter_id_season_idx ON pitches_p2015 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2015_pitch_type_code_idx ON pitches_p2015 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2015_pitcher_id_season_idx ON pitches_p2015 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2016_batter_id_season_idx ON pitches_p2016 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2016_pitch_type_code_idx ON pitches_p2016 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2016_pitcher_id_season_idx ON pitches_p2016 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2017_batter_id_season_idx ON pitches_p2017 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2017_pitch_type_code_idx ON pitches_p2017 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2017_pitcher_id_season_idx ON pitches_p2017 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2018_batter_id_season_idx ON pitches_p2018 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2018_pitch_type_code_idx ON pitches_p2018 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2018_pitcher_id_season_idx ON pitches_p2018 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2019_batter_id_season_idx ON pitches_p2019 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2019_pitch_type_code_idx ON pitches_p2019 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2019_pitcher_id_season_idx ON pitches_p2019 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2020_batter_id_season_idx ON pitches_p2020 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2020_pitch_type_code_idx ON pitches_p2020 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2020_pitcher_id_season_idx ON pitches_p2020 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2021_batter_id_season_idx ON pitches_p2021 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2021_pitch_type_code_idx ON pitches_p2021 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2021_pitcher_id_season_idx ON pitches_p2021 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2022_batter_id_season_idx ON pitches_p2022 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2022_pitch_type_code_idx ON pitches_p2022 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2022_pitcher_id_season_idx ON pitches_p2022 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2023_batter_id_season_idx ON pitches_p2023 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2023_pitch_type_code_idx ON pitches_p2023 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2023_pitcher_id_season_idx ON pitches_p2023 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2024_batter_id_season_idx ON pitches_p2024 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2024_pitch_type_code_idx ON pitches_p2024 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2024_pitcher_id_season_idx ON pitches_p2024 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2025_batter_id_season_idx ON pitches_p2025 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2025_pitch_type_code_idx ON pitches_p2025 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2025_pitcher_id_season_idx ON pitches_p2025 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2026_batter_id_season_idx ON pitches_p2026 USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_p2026_pitch_type_code_idx ON pitches_p2026 USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_p2026_pitcher_id_season_idx ON pitches_p2026 USING btree (pitcher_id, season);

CREATE INDEX IF NOT EXISTS pitches_pdefault_batter_id_season_idx ON pitches_pdefault USING btree (batter_id, season);

CREATE INDEX IF NOT EXISTS pitches_pdefault_pitch_type_code_idx ON pitches_pdefault USING btree (pitch_type_code);

CREATE INDEX IF NOT EXISTS pitches_pdefault_pitcher_id_season_idx ON pitches_pdefault USING btree (pitcher_id, season);

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2009_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2009_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2009_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2009_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2010_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2010_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2010_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2010_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2011_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2011_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2011_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2011_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2012_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2012_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2012_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2012_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2013_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2013_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2013_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2013_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2014_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2014_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2014_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2014_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2015_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2015_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2015_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2015_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2016_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2016_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2016_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2016_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2017_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2017_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2017_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2017_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2018_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2018_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2018_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2018_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2019_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2019_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2019_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2019_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2020_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2020_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2020_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2020_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2021_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2021_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2021_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2021_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2022_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2022_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2022_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2022_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2023_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2023_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2023_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2023_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2024_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2024_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2024_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2024_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2025_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2025_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2025_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2025_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_p2026_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_p2026_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_p2026_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_p2026_pkey;

ALTER INDEX idx_pitches_batter_season ATTACH PARTITION pitches_pdefault_batter_id_season_idx;

ALTER INDEX idx_pitches_ptype_code ATTACH PARTITION pitches_pdefault_pitch_type_code_idx;

ALTER INDEX idx_pitches_pitcher_season ATTACH PARTITION pitches_pdefault_pitcher_id_season_idx;

ALTER INDEX pitches_pk ATTACH PARTITION pitches_pdefault_pkey;

ALTER TABLE ONLY batting
    ADD CONSTRAINT fk_batting_game FOREIGN KEY (game_pk) REFERENCES games(game_pk);

ALTER TABLE ONLY fielding
    ADD CONSTRAINT fk_fielding_game FOREIGN KEY (game_pk) REFERENCES games(game_pk);

ALTER TABLE ONLY linescore
    ADD CONSTRAINT fk_linescore_game FOREIGN KEY (game_pk) REFERENCES games(game_pk);

ALTER TABLE pitches
    ADD CONSTRAINT fk_pitches_away_team FOREIGN KEY (away_team_id) REFERENCES teams(team_id);

ALTER TABLE pitches
    ADD CONSTRAINT fk_pitches_game FOREIGN KEY (game_pk) REFERENCES games(game_pk);

ALTER TABLE pitches
    ADD CONSTRAINT fk_pitches_home_team FOREIGN KEY (home_team_id) REFERENCES teams(team_id);

ALTER TABLE pitches
    ADD CONSTRAINT fk_pitches_venue FOREIGN KEY (venue_id) REFERENCES venues(venue_id);

ALTER TABLE ONLY pitching
    ADD CONSTRAINT fk_pitching_game FOREIGN KEY (game_pk) REFERENCES games(game_pk);

ALTER TABLE ONLY games
    ADD CONSTRAINT games_away_team_id_fkey FOREIGN KEY (away_team_id) REFERENCES teams(team_id);

ALTER TABLE ONLY games
    ADD CONSTRAINT games_home_team_id_fkey FOREIGN KEY (home_team_id) REFERENCES teams(team_id);

ALTER TABLE ONLY games
    ADD CONSTRAINT games_venue_id_fkey FOREIGN KEY (venue_id) REFERENCES venues(venue_id);


CREATE OR REPLACE VIEW betting_game_results_v1 AS
SELECT
    g.game_pk,
    g.game_date,
    g.abstract_game_state,
    g.home_team_id,
    g.away_team_id,
    scores.home_score,
    scores.away_score
FROM games AS g
LEFT JOIN (
    SELECT
        game_pk,
        SUM(runs) FILTER (WHERE team_type = 'home')::integer AS home_score,
        SUM(runs) FILTER (WHERE team_type = 'away')::integer AS away_score
    FROM linescore
    GROUP BY game_pk
) AS scores USING (game_pk);

COMMENT ON VIEW betting_game_results_v1 IS
    'Read-only v1 betting result contract. Consumers must settle only rows whose abstract_game_state is Final and whose scores are non-null.';

CREATE OR REPLACE VIEW betting_player_results_v1 AS
SELECT
    b.game_pk,
    g.game_date,
    g.abstract_game_state,
    b.player_id,
    b.player_name,
    CASE
        WHEN b.gamesplayed = 0 THEN FALSE
        WHEN b.gamesplayed > 0 THEN TRUE
        ELSE NULL
    END AS appeared,
    b.hits,
    b.doubles,
    b.triples,
    b.homeruns AS home_runs,
    b.totalbases AS total_bases,
    b.rbi,
    b.runs,
    b.baseonballs AS walks,
    b.stolenbases AS stolen_bases,
    b.strikeouts
FROM batting AS b
JOIN games AS g USING (game_pk);

COMMENT ON VIEW betting_player_results_v1 IS
    'Read-only v1 betting player result contract. Missing rows and null appeared values are unknown and must remain pending even for Final games.';

COMMENT ON COLUMN betting_player_results_v1.appeared IS
    'Derived from batting.gamesplayed. False is emitted only when the source explicitly stores gamesplayed=0, while null means appearance is unknown and must not void a bet.';
