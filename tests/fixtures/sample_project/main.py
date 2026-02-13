"""Main entry point for sample project."""

from .models import User, Order
from .processor import OrderProcessor, UserProcessor
from .utils import validate_email


def main():
    """Main function demonstrating the application."""
    # Initialize processors
    user_proc = UserProcessor()
    order_proc = OrderProcessor(user_proc)
    
    # Create users
    user1 = user_proc.create_user("alice@example.com", "alice", "smith")
    user2 = user_proc.create_user("bob@example.com", "bob", "jones")
    
    if user1:
        # Create order for user1
        order = order_proc.create_order(user1.id, [19.99, 29.99, 9.99])
        if order:
            total = order_proc.calculate_order_total(order.id)
            print(f"Order {order.id} total: ${total:.2f}")
    
    print("Application completed")


if __name__ == "__main__":
    main()
