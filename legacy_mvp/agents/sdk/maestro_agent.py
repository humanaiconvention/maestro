"""
Agent SDK for Project Maestro

This SDK allows autonomous AI agents to request epistemic grounding,
manage belief distributions, and calculate entropy reduction for settlement.
"""

import requests
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import numpy as np
import uuid

# Setup logger
logger = logging.getLogger(__name__)

@dataclass
class EntropyProof:
    prior_entropy: float
    posterior_entropy: float
    delta_S: float
    is_valid: bool

class MaestroAgent:
    """Gateway for AI agents to participate in the Project Maestro."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str = "http://localhost:8000/api",
        hydra_url: str = "http://127.0.0.1:4444",
        wallet_address: Optional[str] = None,
        timeout: int = 30
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.hydra_url = hydra_url.rstrip("/")
        self.wallet_address = wallet_address or f"0x{uuid.uuid4().hex}"
        self.timeout = timeout
        self.access_token: Optional[str] = None

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Internal helper to handle responses and catch HTTP errors."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.HTTPError:
            return {
                "error": True,
                "status_code": response.status_code,
                "detail": response.text
            }
        except Exception as e:
            return {
                "error": True,
                "status_code": getattr(response, "status_code", 500),
                "detail": str(e)
            }

    def verify_connection(self) -> bool:
        """Pings the base_url /health endpoint to verify connectivity."""
        try:
            # Note: base_url is typically /api, health is usually at root /health
            # But let's try both common patterns.
            root_url = self.base_url.rsplit("/api", 1)[0]
            resp = requests.get(f"{root_url}/health", timeout=self.timeout)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        return False

    def authenticate(self) -> str:
        """Authenticate the AI agent to acquire a scoped machine token."""
        auth_url = f"{self.hydra_url}/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "scope": "offline_access write:grounding_requests"
        }
        auth = (self.client_id, self.client_secret)

        logger.info(f"Authenticating agent '{self.client_id}' with Token Authority...")
        try:
            response = requests.post(auth_url, data=payload, auth=auth, timeout=self.timeout)
            if response.status_code == 200:
                self.access_token = response.json().get("access_token")
                logger.info("Successfully acquired machine token.")
                return self.access_token
        except requests.RequestException as e:
            logger.error(f"Request exception during authentication: {e}")

        logger.warning("Authentication failed or Hydra unreachable. Using mock token for demo.")
        self.access_token = "mock_agent_token_for_local_demo"
        return self.access_token

    def request_grounding(
        self,
        title: str,
        description: str,
        lived_experience_axis: str,
        requested_modality: str,
        protocol_instructions: str,
        estimated_duration_minutes: int,
        compensation_amount: float,
        effective_hourly_estimate: float,
        entropy_reduction_hypothesis: str,
        phase_synchronization_hypothesis: str,
        evidence_artifact_description: str,
        contributions_required: int = 50,
        viability_policy_snapshot: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a formal Request for Grounding matching GroundingRequestCreate schema."""
        if not self.access_token:
            self.authenticate()

        payload = {
            "agent_id": self.client_id,
            "title": title,
            "description": description,
            "lived_experience_axis": lived_experience_axis,
            "requested_modality": requested_modality,
            "protocol_instructions": protocol_instructions,
            "estimated_duration_minutes": estimated_duration_minutes,
            "compensation_amount": compensation_amount,
            "effective_hourly_estimate": effective_hourly_estimate,
            "entropy_reduction_hypothesis": entropy_reduction_hypothesis,
            "phase_synchronization_hypothesis": phase_synchronization_hypothesis,
            "evidence_artifact_description": evidence_artifact_description,
            "contributions_required": contributions_required,
            "viability_policy_snapshot": viability_policy_snapshot or {}
        }

        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.post(
            f"{self.base_url}/grounding-requests", 
            json=payload, 
            headers=headers, 
            timeout=self.timeout
        )
        return self._handle_response(response)

    def evaluate_request(self, request_id: str) -> Dict[str, Any]:
        """Trigger viability evaluation for a grounding request."""
        if not self.access_token:
            self.authenticate()
            
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.post(
            f"{self.base_url}/grounding-requests/{request_id}/evaluate", 
            headers=headers, 
            timeout=self.timeout
        )
        return self._handle_response(response)

    def calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """Calculate Shannon entropy for a discrete probability distribution."""
        probs = np.array(list(distribution.values()))
        probs = probs[probs > 0]
        if not np.isclose(np.sum(probs), 1.0):
            probs = probs / np.sum(probs)

        return float(-np.sum(probs * np.log2(probs)))

    def calculate_entropy_reduction(self, prior: Dict[str, float], posterior: Dict[str, float]) -> EntropyProof:
        """Calculate the reduction in Shannon entropy between prior and posterior beliefs."""
        h_prior = self.calculate_entropy(prior)
        h_post = self.calculate_entropy(posterior)

        delta_s = h_post - h_prior

        return EntropyProof(
            prior_entropy=h_prior,
            posterior_entropy=h_post,
            delta_S=delta_s,
            is_valid=delta_s < 0
        )

    def generate_settlement_proof(
        self,
        prior: Dict[str, float],
        posterior: Dict[str, float],
        observation_count: int,
        total_payout: float
    ) -> Dict[str, Any]:
        """
        Generate proof matching SettlementAttempt Pydantic schema.
        """
        proof = self.calculate_entropy_reduction(prior, posterior)
        claimed_reduction = abs(proof.delta_S) if proof.delta_S < 0 else 0.0

        return {
            "prior_distribution": prior,
            "posterior_distribution": posterior,
            "observation_count": observation_count,
            "claimed_entropy_reduction": claimed_reduction,
            "total_payout": total_payout
        }

    def evaluate_settlement(self, request_id: str, settlement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt thermodynamic settlement for a grounding request."""
        if not self.access_token:
            self.authenticate()
            
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.post(
            f"{self.base_url}/settlements/{request_id}/attempt", 
            json=settlement_data, 
            headers=headers, 
            timeout=self.timeout
        )
        return self._handle_response(response)
