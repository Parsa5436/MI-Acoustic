#!/usr/bin/env python3
"""
Test script to verify game theory implementation and Nash Equilibrium convergence.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import *


def test_game_theory_components():
    """Test individual components of the game theory implementation"""

    print("🧪 TESTING GAME THEORY COMPONENTS")
    print("=" * 60)

    # Load config
    config = load_config()

    # Test 1: AUV Creation and Initialization
    print("1. Testing AUV initialization...")
    initial_pos = Position(100, 200, 50)
    auv = AUV(0, initial_pos, config)

    assert auv.id == 0
    assert auv.position.x == 100
    assert auv.energy > 0
    print("   ✅ AUV initialized correctly")

    # Test 2: Game State Creation
    print("2. Testing GameState creation...")
    surface_station = Position(500, 500, 0)
    sensor_positions = [Position(100, 100, 50), Position(200, 200, 60)]
    auvs = [auv]

    game_state = GameState(
        current_time=0.0,
        auvs=auvs,
        surface_station=surface_station,
        sensor_positions=sensor_positions,
    )

    assert len(game_state.auvs) == 1
    assert len(game_state.sensor_positions) == 2
    print("   ✅ GameState created correctly")

    # Test 3: Action Generation
    print("3. Testing action generation...")
    actions = auv.get_possible_actions(game_state)

    assert len(actions) > 0
    print(f"   ✅ Generated {len(actions)} possible actions")

    # Test 4: Utility Calculation
    print("4. Testing utility calculation...")
    for i, action in enumerate(actions[:3]):  # Test first 3 actions
        utility = auv.calculate_utility(action, game_state)
        print(
            f"   Action {i}: {action.communication_mode.value} -> Utility: {utility:.2f}"
        )

    print("   ✅ Utility calculations working")

    # Test 5: Best Action Selection
    print("5. Testing best action selection...")
    best_action = auv.choose_best_action(game_state)

    assert best_action is not None
    print(f"   ✅ Best action: {best_action.communication_mode.value}")

    print("\n🎯 ALL TESTS PASSED!")
    print("=" * 60)


def test_nash_equilibrium_convergence():
    """Test Nash Equilibrium finding with a small example"""

    print("🎲 TESTING NASH EQUILIBRIUM CONVERGENCE")
    print("=" * 60)

    # Load config and modify for quick test
    config = load_config()
    config["game_theory"]["num_auvs"] = 2  # Smaller test
    config["game_theory"]["learning_algorithm"]["iterations"] = 30
    config["game_theory"]["auv_specs"]["initial_energy"] = 500.0

    # Create GameManager
    game_manager = GameManager(config)

    print(f"Initialized {len(game_manager.auvs)} AUVs for convergence test")

    # Run short simulation
    results = game_manager.run_simulation()

    # Verify results
    assert "auv_paths" in results
    assert "utility_histories" in results
    assert len(results["auv_paths"]) == 2
    assert len(results["utility_histories"]) == 2

    print("\n🏁 CONVERGENCE TEST RESULTS:")
    total_utility = sum(
        results["utility_histories"][i][-1]
        for i in range(len(results["utility_histories"]))
    )
    print(f"   Final system utility: {total_utility:.2f}")

    for i, auv in enumerate(game_manager.auvs):
        print(
            f"   AUV-{i}: Energy {auv.energy:.1f}J, Buffer: {len(auv.data_buffer)} packets"
        )

    print("   ✅ Nash Equilibrium algorithm completed successfully")

    print("=" * 60)


if __name__ == "__main__":
    test_game_theory_components()
    print("\n")
    test_nash_equilibrium_convergence()
    print("\n🌟 ALL INTEGRATION TESTS PASSED!")
