"""Business logic processor."""

from typing import List, Optional

from .models import Order, User
from .utils import calculate_total, format_name, validate_email


class UserProcessor:
    """Process user-related operations."""
    
    def __init__(self):
        """Initialize user processor."""
        self.users: List[User] = []
    
    def create_user(self, email: str, first_name: str, last_name: str) -> Optional[User]:
        """Create a new user.
        
        Args:
            email: User email
            first_name: User's first name
            last_name: User's last name
            
        Returns:
            Created user or None if validation fails
        """
        if not validate_email(email):
            return None
        
        user_id = len(self.users) + 1
        user = User(
            id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        self.users.append(user)
        return user
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None


class OrderProcessor:
    """Process order-related operations."""
    
    def __init__(self, user_processor: UserProcessor):
        """Initialize order processor.
        
        Args:
            user_processor: User processor instance
        """
        self.user_processor = user_processor
        self.orders: List[Order] = []
    
    def create_order(self, user_id: int, items: List[float]) -> Optional[Order]:
        """Create a new order.
        
        Args:
            user_id: ID of user placing order
            items: List of item prices
            
        Returns:
            Created order or None if user not found
        """
        user = self.user_processor.get_user(user_id)
        if not user or not user.is_active:
            return None
        
        order_id = len(self.orders) + 1
        order = Order(id=order_id, user_id=user_id, items=items)
        self.orders.append(order)
        return order
    
    def calculate_order_total(self, order_id: int, tax_rate: float = 0.1) -> Optional[float]:
        """Calculate order total with tax.
        
        Args:
            order_id: Order ID
            tax_rate: Tax rate
            
        Returns:
            Total amount or None if order not found
        """
        for order in self.orders:
            if order.id == order_id:
                return calculate_total(order.items, tax_rate)
        return None
