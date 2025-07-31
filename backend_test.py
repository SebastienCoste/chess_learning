import requests
import sys
import json
from datetime import datetime

class ChessTransformerAPITester:
    def __init__(self, base_url="https://55ba206a-ebfb-4aee-b936-9babc0e8ec39.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            'name': name,
            'success': success,
            'details': details
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            
            success = response.status_code == expected_status
            
            if success:
                try:
                    response_data = response.json()
                    self.log_test(name, True, f"Status: {response.status_code}")
                    return True, response_data
                except json.JSONDecodeError:
                    self.log_test(name, False, f"Invalid JSON response, Status: {response.status_code}")
                    return False, {}
            else:
                try:
                    error_data = response.json()
                    self.log_test(name, False, f"Expected {expected_status}, got {response.status_code}: {error_data}")
                except:
                    self.log_test(name, False, f"Expected {expected_status}, got {response.status_code}: {response.text[:200]}")
                return False, {}

        except requests.exceptions.Timeout:
            self.log_test(name, False, f"Request timeout after {timeout}s")
            return False, {}
        except requests.exceptions.ConnectionError:
            self.log_test(name, False, "Connection error - server may be down")
            return False, {}
        except Exception as e:
            self.log_test(name, False, f"Unexpected error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root endpoint for basic connectivity"""
        success, response = self.run_test(
            "Root Endpoint Connectivity",
            "GET",
            "",
            200
        )
        
        if success:
            # Verify response structure
            expected_keys = ['message', 'model_info']
            if all(key in response for key in expected_keys):
                print(f"   ✓ Response contains expected keys: {expected_keys}")
                if 'device' in response.get('model_info', {}):
                    print(f"   ✓ Model device: {response['model_info']['device']}")
                if 'parameters' in response.get('model_info', {}):
                    print(f"   ✓ Model parameters: {response['model_info']['parameters']:,}")
            else:
                print(f"   ⚠ Missing expected keys in response")
        
        return success

    def test_model_info(self):
        """Test model info endpoint"""
        success, response = self.run_test(
            "Model Info Endpoint",
            "GET",
            "api/model/info",
            200
        )
        
        if success:
            # Verify model info structure
            expected_keys = ['architecture', 'config', 'parameters', 'trainable_parameters']
            if all(key in response for key in expected_keys):
                print(f"   ✓ Model architecture: {response.get('architecture')}")
                print(f"   ✓ Total parameters: {response.get('parameters'):,}")
                print(f"   ✓ Trainable parameters: {response.get('trainable_parameters'):,}")
                
                config = response.get('config', {})
                if 'total_moves' in config:
                    print(f"   ✓ Total possible moves: {config['total_moves']:,}")
                if 'device' in config:
                    print(f"   ✓ Device: {config['device']}")
            else:
                print(f"   ⚠ Missing expected keys in model info")
        
        return success

    def test_move_prediction_default(self):
        """Test move prediction with default starting position"""
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        success, response = self.run_test(
            "Move Prediction - Starting Position",
            "POST",
            "api/predict",
            200,
            data={"fen": starting_fen}
        )
        
        if success:
            # Verify prediction response structure
            expected_keys = ['best_moves', 'position_value', 'legal_moves_count']
            if all(key in response for key in expected_keys):
                print(f"   ✓ Legal moves count: {response.get('legal_moves_count')}")
                print(f"   ✓ Position value: {response.get('position_value'):.4f}")
                
                best_moves = response.get('best_moves', [])
                if best_moves:
                    print(f"   ✓ Top move: {best_moves[0].get('san')} ({best_moves[0].get('move')}) - {best_moves[0].get('probability'):.4f}")
                    print(f"   ✓ Returned {len(best_moves)} move suggestions")
                else:
                    print(f"   ⚠ No moves returned")
            else:
                print(f"   ⚠ Missing expected keys in prediction response")
        
        return success

    def test_move_prediction_midgame(self):
        """Test move prediction with a mid-game position"""
        # Mid-game position after some moves
        midgame_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
        
        success, response = self.run_test(
            "Move Prediction - Mid-game Position",
            "POST",
            "api/predict",
            200,
            data={"fen": midgame_fen}
        )
        
        if success:
            legal_moves_count = response.get('legal_moves_count', 0)
            best_moves = response.get('best_moves', [])
            print(f"   ✓ Mid-game legal moves: {legal_moves_count}")
            if best_moves:
                print(f"   ✓ Top mid-game move: {best_moves[0].get('san')} - {best_moves[0].get('probability'):.4f}")
        
        return success

    def test_move_prediction_endgame(self):
        """Test move prediction with an endgame position"""
        # Simple endgame position
        endgame_fen = "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1"
        
        success, response = self.run_test(
            "Move Prediction - Endgame Position",
            "POST",
            "api/predict",
            200,
            data={"fen": endgame_fen}
        )
        
        if success:
            legal_moves_count = response.get('legal_moves_count', 0)
            best_moves = response.get('best_moves', [])
            print(f"   ✓ Endgame legal moves: {legal_moves_count}")
            if best_moves:
                print(f"   ✓ Top endgame move: {best_moves[0].get('san')} - {best_moves[0].get('probability'):.4f}")
        
        return success

    def test_invalid_fen(self):
        """Test error handling with invalid FEN"""
        invalid_fen = "invalid_fen_string"
        
        success, response = self.run_test(
            "Error Handling - Invalid FEN",
            "POST",
            "api/predict",
            400,
            data={"fen": invalid_fen}
        )
        
        if success:
            print(f"   ✓ Properly rejected invalid FEN with 400 status")
        
        return success

    def test_empty_fen(self):
        """Test error handling with empty FEN"""
        success, response = self.run_test(
            "Error Handling - Empty FEN",
            "POST",
            "api/predict",
            400,
            data={"fen": ""}
        )
        
        if success:
            print(f"   ✓ Properly rejected empty FEN with 400 status")
        
        return success

    def test_training_endpoint(self):
        """Test training endpoint (should accept request but not actually train)"""
        sample_pgn = [
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7"
        ]
        
        success, response = self.run_test(
            "Training Endpoint",
            "POST",
            "api/train",
            200,
            data={
                "pgn_games": sample_pgn,
                "epochs": 5,
                "batch_size": 16
            }
        )
        
        if success:
            expected_keys = ['message', 'epochs', 'batch_size', 'status']
            if all(key in response for key in expected_keys):
                print(f"   ✓ Training request accepted")
                print(f"   ✓ Status: {response.get('status')}")
            else:
                print(f"   ⚠ Missing expected keys in training response")
        
        return success

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Chess Transformer API Tests")
        print(f"📡 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Test basic connectivity first
        if not self.test_root_endpoint():
            print("\n❌ Basic connectivity failed. Stopping tests.")
            return False
        
        # Test model info
        self.test_model_info()
        
        # Test move predictions
        self.test_move_prediction_default()
        self.test_move_prediction_midgame()
        self.test_move_prediction_endgame()
        
        # Test error handling
        self.test_invalid_fen()
        self.test_empty_fen()
        
        # Test training endpoint
        self.test_training_endpoint()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("\n🎉 All tests passed! Backend API is working correctly.")
            return True
        else:
            print(f"\n⚠️  {self.tests_run - self.tests_passed} test(s) failed. Check the details above.")
            
            # Print failed tests
            failed_tests = [test for test in self.test_results if not test['success']]
            if failed_tests:
                print("\n❌ Failed Tests:")
                for test in failed_tests:
                    print(f"   • {test['name']}: {test['details']}")
            
            return False

def main():
    tester = ChessTransformerAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())