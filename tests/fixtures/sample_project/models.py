"""Data models for sample project."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class User:
    """User data model.
    
    Attributes:
        id: Unique user identifier
        email: User email address
        first_name: User's first name
        last_name: User's last name
        is_active: Whether user account is active
    """
    id: int
    email: str
    first_name: str
    last_name: str
    is_active: bool = True
    
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"


@dataclass
class Order:
    """Order data model.
    
    Attributes:
        id: Unique order identifier
        user_id: ID of user who placed order
        items: List of item prices
        status: Order status
    """
    id: int
    user_id: int
    items: List[float]
    status: str = "pending"
    
    def total(self) -> float:
        """Calculate order total."""
        return sum(self.items)
