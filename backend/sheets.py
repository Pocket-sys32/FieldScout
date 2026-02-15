import gspread
from datetime import datetime
import os

# --- SCIENTIFIC NAME LOOKUP (add more later) ---
SCIENTIFIC_NAMES = {
    "beaver": "Castor canadensis",
    "bobcat": "Lynx rufus",
    "coyote": "Canis latrans",
    "striped skunk": "Mephitis mephitis",
    "opossum": "Didelphis virginiana",
    "bt deer": "Odocoileus hemionus columbianus", # Black-tailed Deer
    "gray fox": "Urocyon cinereoargenteus",
    "raccoon": "Procyon lotor",
    "desert cottontail": "Sylvilagus audubonii",
    "fox squirrel": "Sciurus niger",
    "ca ground squirrel": "Otospermophilus beecheyi",
    "ca quail": "Callipepla californica",
    "golden-crown sparrow": "Zonotrichia atricapilla",
    "wild turkey": "Meleagris gallopavo",
    "river otter": "Lontra canadensis",
    "ca scrub jay": "Aphelocoma californica",
    "american badger": "Taxidea taxus",
    "ca towhee": "Melozone crissalis",
    "northern mockingbird": "Mimus polyglottos",
    "anna's hummingbird": "Calypte anna",
    "raptor": "Raptor sp.",                # General term for birds of prey
    "frog sp.": "Anura sp."                # General term for frogs/toads
}


class SheetLogger:
    def __init__(self, json_keyfile, sheet_name):
        try:
            if not os.path.exists(json_keyfile):
                print(f"❌ Error: Could not find {json_keyfile}")
                self.sheet = None
                return

            gc = gspread.service_account(filename=json_keyfile)
            self.sheet = gc.open(sheet_name).sheet1
            print(f"✅ Connected to Google Sheet: {sheet_name}")

        except Exception as e:
            print(f"❌ Error connecting to Sheets: {e}")
            self.sheet = None

    def log_detection(self, species, confidence, filename, date_str=None, time_str=None):
        if not self.sheet:
            return


        now = datetime.now()

        final_date = date_str if date_str else now.strftime("%Y-%m-%d")
        final_time = time_str if time_str else now.strftime("%H:%M:%S")

        # 2. Get Scientific Name
        sc_name = SCIENTIFIC_NAMES.get(species.lower(), "Unknown Species")

        # 3. Create Row Data
        row_data = [
            final_date,  # Col A: Date (From Video)
            final_time,  # Col B: Time (From Video)
            species,  # Col C
            sc_name,  # Col D
            1,  # Col E
            filename,  # Col F
            f"Confidence: {confidence:.0%}"  # Col G
        ]

        try:
            col_a_values = self.sheet.col_values(1)
            next_row = len(col_a_values) + 1

            cell_range = f"A{next_row}:G{next_row}"
            self.sheet.update(range_name=cell_range, values=[row_data])
            print(f"📝 Logged to Row {next_row}: {species} at {final_date} {final_time}")

        except Exception as e:
            print(f"⚠️ Failed to write row: {e}")