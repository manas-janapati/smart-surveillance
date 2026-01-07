import csv
from datetime import datetime
from bisect import bisect_left


class GPSReader:
    def __init__(self, gps_csv_path):
        self.timestamps = []
        self.coords = []

        with open(gps_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = datetime.fromisoformat(
                    row["timestamp"].replace("Z", "+00:00")
                )
                self.timestamps.append(ts)
                self.coords.append(
                    (float(row["lat"]), float(row["lon"]))
                )

        if not self.timestamps:
            raise ValueError("GPS log is empty")

    def get_closest(self, timestamp):
        """
        Return (lat, lon) for the closest GPS timestamp
        """
        idx = bisect_left(self.timestamps, timestamp)

        if idx == 0:
            return self.coords[0]
        if idx >= len(self.timestamps):
            return self.coords[-1]

        before = self.timestamps[idx - 1]
        after = self.timestamps[idx]

        if (timestamp - before) <= (after - timestamp):
            return self.coords[idx - 1]
        else:
            return self.coords[idx]
