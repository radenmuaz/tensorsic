import sys
from . import engine  # Import from the package

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tensorsic <inputfile>")
        sys.exit(1)

    input_file = sys.argv[1]
    print(f"Executing lib as a package with input file: {input_file}")
    # Optionally delegate to my_module
    engine.run()

if __name__ == "__main__":
    main()
