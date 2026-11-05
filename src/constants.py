"""Shared constants for leak-free logistic features."""
FEATURE_COLS = [
    "HOME_ELO", "AWAY_ELO", "ELO_DIFF", "ELO_HOME_WIN_PROB",
    "HOME_REST", "AWAY_REST", "REST_DIFF",
    "HOME_PRIOR_PPG", "AWAY_PRIOR_PPG",
    "HOME_PRIOR_OPP_PPG", "AWAY_PRIOR_OPP_PPG",
    "HOME_PRIOR_NET", "AWAY_PRIOR_NET",
    "PPG_DIFF", "NET_DIFF", "IMPLIED_MARGIN",
]
TRAIN_SEASONS = ["2023-24", "2024-25", "2025-26"]
LIVE_SEASON = "2026-27"
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME = 50.0
ROLL = 10

# ESPN abbreviation → NBA stats abbreviation
ESPN_TO_NBA = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",
    "UTAH": "UTA", "WSH": "WAS",
}
