def rag_response(text, cat, prio, sentiment):
    solutions = {
        "Electronics": "Please try resetting your device. If the issue persists, we can arrange a technical inspection.",
        "Billing": "Your refund is being processed and should appear in your account within 5-7 business days.",
        "Delivery": "Your order is with our courier and will be delivered shortly. Track it via the link in your email.",
        "Household": "For assembly issues, please refer to the digital manual on our website or request a technician.",
        "Clothing": "Returns are accepted within 14 days if the tags are still attached and the item is unworn."
    }
    
    response = solutions.get(cat, "Our support team has received your request and will contact you shortly.")
    
    if sentiment.lower() == "negative":
        return f"We apologize for the trouble. {response}"
    return f"We're happy to help! {response}"