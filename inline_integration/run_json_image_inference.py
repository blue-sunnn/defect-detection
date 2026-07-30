import sys
import json

from model_integration import predict

def main():

    input_data = sys.stdin.read()

    if not input_data:
            sys.stderr.write("No data received from C#")
            return
    
    try:
        records = json.loads(input_data)
        response = predict(records)
    except Exception as e:
            sys.stderr.write(f"Python Processing Error: {str(e)}")
            
    print(json.dumps(response, indent=4))


if __name__ == "__main__":
    main()
