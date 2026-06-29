"""
Sprint 1 – Phase 4: Testing
Tests all 3 required test cases from the problem statement.
Run: python test_extractor.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from parser import parse_llm_output, EventInfo

# ── We unit-test the parser with mock LLM responses ──────────────────────────
# (No real API call needed for unit tests)

def test_complete_event():
    """Test Case 1: Complete event description → all fields extracted."""
    mock_llm_response = """{
        "event_name": "AI Conference",
        "event_date": "July 10",
        "event_time": "10 AM",
        "event_location": "Hyderabad",
        "organizer": "TechNova"
    }"""
    result = parse_llm_output(mock_llm_response)
    assert result.event_name == "AI Conference"
    assert result.event_date == "July 10"
    assert result.event_time == "10 AM"
    assert result.event_location == "Hyderabad"
    assert result.organizer == "TechNova"
    print("✅ Test 1 PASSED: Complete event — all fields extracted correctly")
    return result

def test_missing_organizer():
    """Test Case 2: Missing organizer → organizer should be 'not_available'."""
    mock_llm_response = """{
        "event_name": "Annual Tech Summit",
        "event_date": "August 5th",
        "event_time": "3 PM",
        "event_location": "Grand Hyatt, Bangalore",
        "organizer": "not_available"
    }"""
    result = parse_llm_output(mock_llm_response)
    assert result.organizer == "not_available", f"Expected 'not_available', got '{result.organizer}'"
    assert result.event_name != "not_available"
    print("✅ Test 2 PASSED: Missing organizer → 'not_available'")
    return result

def test_missing_time_and_location():
    """Test Case 3: Missing time or location → fields marked clearly."""
    mock_llm_response = """{
        "event_name": "Web3 Hackathon",
        "event_date": "September 20th",
        "event_time": "not_available",
        "event_location": "not_available",
        "organizer": "BlockBuilders"
    }"""
    result = parse_llm_output(mock_llm_response)
    assert result.event_time == "not_available"
    assert result.event_location == "not_available"
    print("✅ Test 3 PASSED: Missing time & location → both 'not_available'")
    return result

def test_json_with_markdown_fence():
    """Edge case: LLM wraps response in markdown code block."""
    mock_llm_response = """```json
{
    "event_name": "DevFest",
    "event_date": "October 1",
    "event_time": "9 AM",
    "event_location": "Chennai",
    "organizer": "GDG Chennai"
}
```"""
    result = parse_llm_output(mock_llm_response)
    assert result.event_name == "DevFest"
    print("✅ Test 4 PASSED: Markdown fenced JSON → parsed correctly")
    return result

def test_empty_fields_default_to_na():
    """Edge case: LLM returns empty string for missing fields."""
    mock_llm_response = """{
        "event_name": "Mystery Event",
        "event_date": "",
        "event_time": null,
        "event_location": "",
        "organizer": ""
    }"""
    result = parse_llm_output(mock_llm_response)
    assert result.event_date == "not_available"
    assert result.event_location == "not_available"
    print("✅ Test 5 PASSED: Empty/null fields → defaulted to 'not_available'")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("   EVENT EXTRACTOR — TEST SUITE")
    print("=" * 50 + "\n")

    results = []
    tests = [
        test_complete_event,
        test_missing_organizer,
        test_missing_time_and_location,
        test_json_with_markdown_fence,
        test_empty_fields_default_to_na,
    ]

    passed = 0
    for test_fn in tests:
        try:
            results.append(test_fn())
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test_fn.__name__}: {e}")
        except Exception as e:
            print(f"💥 ERROR in {test_fn.__name__}: {e}")

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{len(tests)} tests passed")
    print("=" * 50 + "\n")
