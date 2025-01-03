import os
import csv


def add_to_experiments(metrics):
  filename = "Experiments.csv"
  file_exists = os.path.isfile(filename)

  # Read existing data and headers
  if file_exists:
    with open(filename, mode="r") as check_file:
      reader = csv.DictReader(check_file, quotechar='"', quoting=csv.QUOTE_ALL)
      rows = list(reader)
      existing_headers = reader.fieldnames
  else:
    rows = []
    existing_headers = list(metrics.keys())

  # Merge columns and update rows
  all_headers = list(set(existing_headers).union(set(metrics.keys())))
  for row in rows:
    for key in all_headers:
      if key not in row:
        row[key] = None

  # Append the new entry
  new_entry = {key: metrics.get(key, None) for key in all_headers}
  rows.append(new_entry)

  # Write the merged data back to the CSV
  with open(filename, mode="w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=all_headers, quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)


def add_to_trials(trials):
  results_file = "Trials.csv"
  if os.path.isfile(results_file):
    trials.to_csv(results_file, mode="a", header=False, index=False)
  else:
    trials.to_csv(results_file, mode="w", index=False)
