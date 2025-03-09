from typing import Dict, Any


def augment_with_path_references_scope(section: Dict[str, Any]) -> str:
    """Augment texts with additional context before embedding."""
    return f"""{' > '.join(section['path'] + [section['title']])}
{section['text']}

References: {', '.join(section['references'])}
Scope: {section['scope']}
"""

def augment_with_references_scope(section: Dict[str, Any]) -> str:
    """Augment texts with additional context before embedding."""
    return f"""References: {', '.join(section['references'])}
Scope: {section['scope']}
{section['text']}"""

def augment_with_title(section: Dict[str, Any]) -> str:
    """Augment texts with additional context before embedding."""
    return f"""{section['title']}
{section['text']}"""