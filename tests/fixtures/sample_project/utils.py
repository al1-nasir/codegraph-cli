"""Utility functions for sample project."""


def validate_email(email: str) -> bool:
    """Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    return "@" in email and "." in email.split("@")[1]


def format_name(first: str, last: str) -> str:
    """Format full name from first and last name.
    
    Args:
        first: First name
        last: Last name
        
    Returns:
        Formatted full name
    """
    return f"{first.capitalize()} {last.capitalize()}"


def calculate_total(items: list, tax_rate: float = 0.1) -> float:
    """Calculate total with tax.
    
    Args:
        items: List of item prices
        tax_rate: Tax rate as decimal (default 0.1 for 10%)
        
    Returns:
        Total amount including tax
    """
    subtotal = sum(items)
    tax = subtotal * tax_rate
    return subtotal + tax
