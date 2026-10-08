import pandas as pd
from acnportal.acndata import DataClient

# ==========================================
# 1. ENTER YOUR ACN API TOKEN
# ==========================================
TOKEN = "Afl0UX7i4xrvbpdF8B6YWlH-CcAI5t0-JLrP6lnNAz8"

client = DataClient(TOKEN)

# ==========================================
# 2. STUDY PERIOD
#    April 1, 2019 through May 31, 2019
# ==========================================
start = "Mon, 1 Apr 2019 00:00:00 GMT"
end   = "Sat, 1 Jun 2019 00:00:00 GMT"

condition = (
    f'connectionTime >= "{start}" '
    f'and connectionTime < "{end}"'
)

# ==========================================
# 3. DOWNLOAD EACH SITE
# ==========================================
sites = {
    "caltech": "caltech_sessions.csv",
    "jpl": "jpl_sessions.csv",
    "office001": "office001_sessions.csv"
}

for site, filename in sites.items():

    print(f"\nDownloading {site}...")

    sessions = list(
        client.get_sessions(
            site=site,
            cond=condition,
            sort="connectionTime"
        )
    )

    df = pd.DataFrame(sessions)

    df.to_csv(filename, index=False)

    print(f"Downloaded: {len(df)} sessions")
    print(f"Saved as: {filename}")

print("\n ALL 3 SITES DOWNLOADED!")