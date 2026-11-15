"""
tests/test_llm_advisor.py — Unit tests for GreenSense Horticultural AI Suite
-----------------------------------------------------------------------------
Tests offline expert agronomic reasoning, loss incident diagnostics, SOP action
plans, executive audit narrative generation, dynamic query expansion, visual
plant health diagnostics, and Copilot Q&A reasoning.
"""

try:
    import pytest
    fixture_decorator = pytest.fixture
except Exception:
    def fixture_decorator(fn):
        return fn

from llm_advisor import (
    ASSET_ESTIMATED_REPLACEMENT_COST_INR,
    EMBASSY_SOP_RULES,
    ZONE_PROFILES,
    LLMAdvisor,
)


@fixture_decorator
def advisor():
    """Provides an isolated offline LLM Advisor instance."""
    return LLMAdvisor(provider="offline")


def test_advisor_initialization(advisor):
    """Verifies default provider and model configuration."""
    assert advisor.provider == "offline"
    assert "greensense" in advisor.model_name.lower() or "offline" in advisor.model_name.lower()


def test_diagnose_loss_critical(advisor):
    """Verifies critical loss diagnosis when asset count drops by 50% or more."""
    diag = advisor.diagnose_loss(
        zone="Manyata Block D Garden",
        asset_label="a tree",
        latest_count=4,
        rolling_avg=12.0,
        drop_pct=66.7,
    )
    assert diag["severity"] == "CRITICAL"
    assert "66.7% drop" in diag["drop_magnitude"]
    assert len(diag["root_causes"]) >= 3
    assert "immediate_24h" in diag["sop_action_plan"]
    assert "24 hours" in diag["sop_action_plan"]["immediate_24h"]
    assert "corrective_7d" in diag["sop_action_plan"]
    assert "monitoring_30d" in diag["sop_action_plan"]
    assert "₹" in diag["estimated_liability_inr"]
    assert "SOP §4.2" in diag["sop_reference"]


def test_diagnose_loss_high_severity(advisor):
    """Verifies high loss diagnosis when asset count drops between 30% and 50%."""
    diag = advisor.diagnose_loss(
        zone="Embassy GolfLinks Zone 1",
        asset_label="a palm tree",
        latest_count=7,
        rolling_avg=11.0,
    )
    assert diag["severity"] == "HIGH"
    assert "₹" in diag["estimated_liability_inr"]
    assert any("palm" in cause.lower() or "weevil" in cause.lower() or "salinity" in cause.lower() for cause in diag["root_causes"])


def test_diagnose_loss_normal_variation(advisor):
    """Verifies normal variation classification when count drop is below 15%."""
    diag = advisor.diagnose_loss(
        zone="Boulevard West Lawn",
        asset_label="a flowering shrub",
        latest_count=18,
        rolling_avg=20.0,
        drop_pct=10.0,
    )
    assert diag["severity"] == "NORMAL_VARIATION"


def test_generate_audit_narrative(advisor):
    """Verifies generation of formal SME Landscaping Audit memorandum."""
    summary_data = {
        "Manyata Block D Garden": {"a tree": 8, "a palm tree": 14, "a flowering shrub": 22},
        "Embassy GolfLinks Zone 1": {"a palm tree": 12, "a lawn / grass area": 5},
    }
    alerts = ["⚠ Tree count in Manyata dropped from 12.0 to 8"]

    narrative = advisor.generate_audit_narrative(summary_data=summary_data, alerts=alerts)
    assert "# GreenSense SME Landscaping Audit" in narrative
    assert "Manyata Block D Garden" in narrative
    assert "Embassy GolfLinks Zone 1" in narrative
    assert "ATTENTION REQUIRED" in narrative
    assert "SOP §4.2" in narrative


def test_generate_audit_narrative_zero_alerts(advisor):
    """Verifies compliance status messaging when no alerts exist."""
    summary_data = {
        "Boulevard West Lawn": {"a palm tree": 10, "a flowering shrub": 15},
    }
    narrative = advisor.generate_audit_narrative(summary_data=summary_data, alerts=[])
    assert "COMPLIANCE STATUS" in narrative
    assert "Zero critical loss incidents flagged" in narrative


def test_expand_detection_queries_tropical(advisor):
    """Verifies open-vocabulary prompt expansion for tropical palm and flower query."""
    res = advisor.expand_detection_queries("tropical poolside garden with royal palms and bougainvillea")
    assert "expanded_queries" in res
    assert "category_mapping" in res
    assert len(res["expanded_queries"]) >= 3
    # Check that canonical categories map back
    assert any("palm" in q for q in res["expanded_queries"])
    assert any(cat == "a palm tree" for cat in res["category_mapping"].values())


def test_expand_detection_queries_empty_fallback(advisor):
    """Verifies graceful fallback to default queries on empty prompt."""
    res = advisor.expand_detection_queries("")
    assert len(res["expanded_queries"]) >= 6
    assert "a tree" in res["expanded_queries"]


def test_diagnose_plant_health(advisor):
    """Verifies visual health diagnostics for palms and shrubs."""
    palm_health = advisor.diagnose_plant_health(asset_label="a palm tree", zone="Manyata Block D Garden")
    assert palm_health["status"] != ""
    assert palm_health["health_score"] >= 0.8
    assert "foliar_condition" in palm_health
    assert "sop_action" in palm_health

    shrub_health = advisor.diagnose_plant_health(asset_label="a flowering shrub", zone="Boulevard West Lawn")
    assert shrub_health["health_score"] >= 0.8


def test_copilot_loss_query(advisor):
    """Verifies Copilot answers questions regarding loss alerts and root causes."""
    reg_rows = [{"zone": "Manyata Block D Garden", "label": "a tree", "count": 7}]
    ans = advisor.ask_copilot("Which zones have active tree loss alerts and why?", register_rows=reg_rows)
    assert "Manyata Block D Garden" in ans
    assert "SOP §4.2" in ans or "41.7%" in ans


def test_copilot_inventory_total_query(advisor):
    """Verifies Copilot summarizes total asset inventory."""
    summary_data = {
        "Manyata Block D Garden": {"a tree": 10, "a palm tree": 5},
        "Embassy GolfLinks Zone 1": {"a palm tree": 15},
    }
    ans = advisor.ask_copilot("What is our total living plant count?", summary_data=summary_data)
    assert "30" in ans or "Total Verified Flora" in ans


def test_copilot_sop_rules_query(advisor):
    """Verifies Copilot answers questions on STP water TDS and watering schedule."""
    ans = advisor.ask_copilot("What are the SOP rules for STP water salinity TDS?")
    assert "1200 ppm" in ans
    assert "Summer Protocol" in ans


def test_copilot_warranty_query(advisor):
    """Verifies Copilot answers questions on contractor replacement SLA."""
    ans = advisor.ask_copilot("What is the contractor warranty replacement SLA?")
    assert "7 working days" in ans or "7-day" in ans
    assert "₹" in ans


def test_cloud_fallback_on_unauthenticated(advisor):
    """Verifies that an unauthenticated or invalid cloud provider gracefully falls back without crashing."""
    cloud_advisor = LLMAdvisor(provider="gemini", api_key="INVALID_TEST_KEY")
    diag = cloud_advisor.diagnose_loss(
        zone="Manyata Block D Garden",
        asset_label="a tree",
        latest_count=6,
        rolling_avg=12.0,
    )
    # Should fall back cleanly to offline synthesis without throwing an exception
    assert diag["severity"] in ("CRITICAL", "HIGH")
    assert len(diag["root_causes"]) >= 3
