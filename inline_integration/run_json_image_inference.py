import json

from model_integration import predict

# Sample request in the exact JSON shape the external GUI sends: a list of groups, each with
# an "Images" list of {"preparedImagePath": <windows path>} entries. Used here as a local
# smoke test for model_integration.predict without needing the real GUI process.
input_path = r'''
[
  {
    "Images": [
      {
        "preparedImagePath": "C:\\path\\to\\images\\sample_short.jpg"
      },
      {
        "preparedImagePath": "C:\\path\\to\\images\\sample_protrusion.jpg"
      },
      {
        "preparedImagePath": "C:\\path\\to\\images\\sample_open.jpg"
      },
      {
        "preparedImagePath": "C:\\path\\to\\images\\sample_pinhole.jpg"
      },
      {
        "preparedImagePath": "C:\\path\\to\\images\\sample_short_2.jpg"
      },
      {
        "preparedImagePath": "C:\\path\\to\\images\\sample_mousebite.jpg"
      }
    ]
  }
]
'''

def main():
    try:
        request = json.loads(input_path)
        response = predict(request)
    except Exception:
        # Only reachable if the hardcoded input_path above is bad JSON, or predict() itself
        # raises unexpectedly (it's designed not to) — fall back to an empty result set.
        response = {
            "Results": [],
        }
    print(json.dumps(response, indent=4))


if __name__ == "__main__":
    main()
