#!/usr/bin/env python3
"""
Chess LLM Setup and Runner Script
Automates the entire pipeline: download data, prepare training data, train model, and play.
"""

import subprocess
import sys
import os
from pathlib import Path
import time


class ChessLLMSetup:
    def __init__(self):
        self.scripts = {
            'download': '1_download_chess_data.py',
            'prepare': '2_prepare_training_data.py',
            'train': '3_train_chess_llm.py',
            'play': '4_play_chess.py'
        }

    def check_requirements(self):
        """Check if all required files exist"""
        print("Checking setup...")

        missing_files = []
        for script_name, script_file in self.scripts.items():
            if not Path(script_file).exists():
                missing_files.append(script_file)

        if not Path('requirements.txt').exists():
            missing_files.append('requirements.txt')

        if missing_files:
            print(f"❌ Missing files: {', '.join(missing_files)}")
            return False

        print("✅ All required files found")
        return True

    def install_dependencies(self):
        """Install Python dependencies"""
        print("\n📦 Installing dependencies...")

        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Error installing dependencies: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False

    def run_script(self, script_name, script_file):
        """Run a Python script and show progress"""
        print(f"\n🚀 Running {script_name}...")
        print(f"Command: python {script_file}")
        print("-" * 50)

        try:
            # Run the script and show output in real-time
            process = subprocess.Popen([
                sys.executable, script_file
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1)

            # Show output line by line
            for line in process.stdout:
                print(line, end='')

            process.wait()

            if process.returncode == 0:
                print(f"✅ {script_name} completed successfully")
                return True
            else:
                print(f"❌ {script_name} failed with return code {process.returncode}")
                return False

        except Exception as e:
            print(f"❌ Error running {script_name}: {e}")
            return False

    def check_outputs(self, step):
        """Check if expected output files exist"""
        expected_files = {
            'download': ['chess_games_10k.pgn'],
            'prepare': ['chess_train.jsonl', 'chess_val.jsonl'],
            'train': ['chess-mistral-7b-lora/adapter_config.json']
        }

        if step in expected_files:
            for file_path in expected_files[step]:
                if Path(file_path).exists():
                    print(f"✅ Output file created: {file_path}")
                else:
                    print(f"⚠️  Expected output file missing: {file_path}")

    def estimate_time(self, step):
        """Estimate time for each step"""
        time_estimates = {
            'download': "5-15 minutes",
            'prepare': "2-10 minutes",
            'train': "2-8 hours",
            'play': "Interactive"
        }
        return time_estimates.get(step, "Unknown")

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("🏁 Starting Chess LLM Full Pipeline")
        print("=" * 60)

        steps = [
            ('download', 'Downloading Chess Data'),
            ('prepare', 'Preparing Training Data'),
            ('train', 'Training the Model'),
        ]

        start_time = time.time()

        for step, description in steps:
            print(f"\n📋 Step: {description}")
            print(f"⏱️  Estimated time: {self.estimate_time(step)}")

            input(f"Press Enter to start {step} (or Ctrl+C to cancel)...")

            step_start = time.time()
            success = self.run_script(description, self.scripts[step])
            step_duration = time.time() - step_start

            print(f"⏱️  Step completed in {step_duration / 60:.1f} minutes")

            if success:
                self.check_outputs(step)
            else:
                print(f"❌ Pipeline failed at step: {description}")
                return False

        total_duration = time.time() - start_time
        print(f"\n🎉 Full pipeline completed in {total_duration / 60:.1f} minutes!")
        print("\n🎮 Ready to play! Run: python play_chess.py")

        return True

    def interactive_menu(self):
        """Show interactive menu for running individual steps"""
        while True:
            print("\n🏆 Chess LLM Training Pipeline")
            print("=" * 40)
            print("1. Install Dependencies")
            print("2. Download Chess Data")
            print("3. Prepare Training Data")
            print("4. Train Model")
            print("5. Play Chess")
            print("6. Run Full Pipeline")
            print("7. Check Status")
            print("0. Exit")

            choice = input("\nEnter your choice (0-7): ").strip()

            if choice == '0':
                print("Goodbye! 👋")
                break
            elif choice == '1':
                self.install_dependencies()
            elif choice == '2':
                self.run_script("Data Download", self.scripts['download'])
                self.check_outputs('download')
            elif choice == '3':
                self.run_script("Data Preparation", self.scripts['prepare'])
                self.check_outputs('prepare')
            elif choice == '4':
                print(f"⏱️  Training will take approximately {self.estimate_time('train')}")
                confirm = input("Continue? (y/n): ").lower()
                if confirm == 'y':
                    self.run_script("Model Training", self.scripts['train'])
                    self.check_outputs('train')
            elif choice == '5':
                print("🎮 Starting chess game...")
                self.run_script("Chess Game", self.scripts['play'])
            elif choice == '6':
                confirm = input("Run full pipeline? This will take several hours (y/n): ").lower()
                if confirm == 'y':
                    self.run_full_pipeline()
            elif choice == '7':
                self.check_project_status()
            else:
                print("Invalid choice. Please try again.")

    def check_project_status(self):
        """Check the current status of the project"""
        print("\n📊 Project Status")
        print("-" * 30)

        # Check data files
        data_files = {
            'Chess Data (PGN)': 'chess_games_10k.pgn',
            'Training Data': 'chess_train.jsonl',
            'Validation Data': 'chess_val.jsonl'
        }

        print("Data Files:")
        for name, file_path in data_files.items():
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                print(f"  ✅ {name}: {file_path} ({size:.1f} MB)")
            else:
                print(f"  ❌ {name}: {file_path} (missing)")

        # Check model
        model_dir = Path('chess-mistral-7b-lora')
        print(f"\nTrained Model:")
        if model_dir.exists():
            model_files = list(model_dir.glob('*'))
            print(f"  ✅ Model directory: {model_dir} ({len(model_files)} files)")
            for file in model_files:
                print(f"    - {file.name}")
        else:
            print(f"  ❌ Model directory: {model_dir} (missing)")

        # Check scripts
        print(f"\nScripts:")
        for name, file_path in self.scripts.items():
            if Path(file_path).exists():
                print(f"  ✅ {name}: {file_path}")
            else:
                print(f"  ❌ {name}: {file_path}")


def main():
    """Main function"""
    setup = ChessLLMSetup()

    # Check if all files exist
    if not setup.check_requirements():
        print("\n❌ Setup incomplete. Please ensure all files are in the current directory.")
        return

    print("\n🎯 Chess LLM Setup Ready!")

    # Show menu
    try:
        setup.interactive_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted. Goodbye!")


if __name__ == "__main__":
    main()