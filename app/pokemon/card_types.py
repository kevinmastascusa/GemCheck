"""
Pokémon card type definitions, rarities, and card part classifications.
Specialized for accurate Pokémon TCG grading analysis.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import re


class PokemonCardType(Enum):
    """Types of Pokémon cards with specific grading considerations."""
    POKEMON = "pokemon"
    TRAINER = "trainer" 
    ENERGY = "energy"
    SPECIAL_ENERGY = "special_energy"
    
    # Specific Pokémon subtypes
    BASIC = "basic"
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    MEGA = "mega"
    GX = "gx"
    EX = "ex"
    V = "v"
    VMAX = "vmax"
    VSTAR = "vstar"
    RADIANT = "radiant"
    AMAZING_RARE = "amazing_rare"
    
    # Trainer subtypes
    SUPPORTER = "supporter"
    ITEM = "item"
    STADIUM = "stadium"
    POKEMON_TOOL = "pokemon_tool"


class PokemonRarity(Enum):
    """Pokémon card rarities with grading impact factors."""
    # Common rarities
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    
    # Holographic rarities
    RARE_HOLO = "rare_holo"
    REVERSE_HOLO = "reverse_holo"
    
    # Special rarities
    ULTRA_RARE = "ultra_rare"
    SECRET_RARE = "secret_rare"
    RAINBOW_RARE = "rainbow_rare"
    GOLD_RARE = "gold_rare"
    
    # Promotional
    PROMO = "promo"
    PROMO_BLACK_STAR = "promo_black_star"
    
    # Vintage specific
    FIRST_EDITION = "first_edition"
    SHADOWLESS = "shadowless"
    BASE_SET_2 = "base_set_2"
    
    # Modern special
    AMAZING_RARE = "amazing_rare"
    RADIANT_RARE = "radiant_rare"
    ALTERNATE_ART = "alternate_art"
    FULL_ART = "full_art"
    TEXTURED = "textured"


class PokemonCardEra(Enum):
    """Pokémon card eras with different grading standards."""
    VINTAGE = "vintage"  # Base Set - Neo Destiny (1998-2001)
    E_CARD = "e_card"    # E-Card series (2001-2003)
    EX = "ex"            # EX series (2003-2007)
    DIAMOND_PEARL = "diamond_pearl"  # DP series (2007-2011)
    BLACK_WHITE = "black_white"      # BW series (2011-2014)
    XY = "xy"            # XY series (2014-2017)
    SUN_MOON = "sun_moon"  # SM series (2017-2019)
    SWORD_SHIELD = "sword_shield"  # SWSH series (2020-2022)
    SCARLET_VIOLET = "scarlet_violet"  # SV series (2023+)


@dataclass
class PokemonCardParts:
    """Definitions of Pokémon card parts for analysis."""
    # Core card elements
    artwork: str = "Pokemon artwork/illustration"
    name: str = "Pokemon/card name"
    hp: str = "Hit points (HP)"
    type_symbols: str = "Pokemon type symbols/energy symbols"
    
    # Card borders and frames
    yellow_border: str = "Yellow card border (vintage)"
    silver_border: str = "Silver card border (modern)"
    card_frame: str = "Card frame around artwork"
    text_box: str = "Attack/ability text box"
    
    # Rarity and set information
    rarity_symbol: str = "Rarity symbol (circle, diamond, star, etc.)"
    set_symbol: str = "Set symbol/logo"
    card_number: str = "Card number (e.g., 150/150)"
    copyright_line: str = "Copyright information"
    
    # Special elements
    holographic_foil: str = "Holographic foil pattern"
    texture: str = "Card texture (smooth, textured, etc.)"
    energy_symbols: str = "Energy requirement symbols"
    weakness_resistance: str = "Weakness and resistance symbols"
    retreat_cost: str = "Retreat cost symbols"
    
    # Modern card elements
    regulation_mark: str = "Regulation mark (letter in bottom left)"
    qr_code: str = "QR code (modern cards)"
    authenticity_seal: str = "Security features/holo pattern"


@dataclass
class GradingCriteria:
    """Pokémon-specific grading criteria and weight adjustments."""
    card_type: PokemonCardType
    rarity: PokemonRarity
    era: PokemonCardEra
    
    # Weight adjustments based on card characteristics
    centering_weight: float = 35.0
    edges_weight: float = 20.0
    corners_weight: float = 20.0
    surface_weight: float = 25.0
    
    # Special considerations
    holographic_penalty_factor: float = 1.0  # Holos judged more strictly
    vintage_tolerance: float = 1.0  # Vintage cards may have different standards
    print_quality_factor: float = 1.0  # Some sets have known print issues


class PokemonCardDetector:
    """Specialized detector for Pokémon card types and characteristics."""
    
    def __init__(self):
        self.type_keywords = {
            PokemonCardType.POKEMON: [
                "HP", "retreat cost", "weakness", "resistance", "evolves from"
            ],
            PokemonCardType.TRAINER: [
                "trainer", "supporter", "item", "stadium", "pokemon tool"
            ],
            PokemonCardType.ENERGY: [
                "basic energy", "provides", "energy"
            ]
        }
        
        self.rarity_indicators = {
            PokemonRarity.COMMON: ["●", "circle", "solid circle"],
            PokemonRarity.UNCOMMON: ["♦", "diamond", "solid diamond"],
            PokemonRarity.RARE: ["★", "star", "hollow star"],
            PokemonRarity.RARE_HOLO: ["★", "holographic", "holo pattern"],
            PokemonRarity.ULTRA_RARE: ["ultra rare", "UR"],
            PokemonRarity.SECRET_RARE: ["secret rare", "SR", "rainbow"],
            PokemonRarity.PROMO: ["promo", "black star promo"]
        }
        
        self.era_indicators = {
            PokemonCardEra.VINTAGE: [
                "1998", "1999", "2000", "2001", "wizards of the coast",
                "shadowless", "first edition", "base set"
            ],
            PokemonCardEra.E_CARD: ["e-card", "2001", "2002", "2003"],
            PokemonCardEra.EX: ["ex", "2003", "2004", "2005", "2006", "2007"],
            PokemonCardEra.DIAMOND_PEARL: ["dp", "diamond", "pearl", "platinum"],
            PokemonCardEra.BLACK_WHITE: ["bw", "black & white"],
            PokemonCardEra.XY: ["xy", "kalos"],
            PokemonCardEra.SUN_MOON: ["sm", "alola", "sun & moon"],
            PokemonCardEra.SWORD_SHIELD: ["swsh", "galar", "sword & shield"],
            PokemonCardEra.SCARLET_VIOLET: ["sv", "paldea", "scarlet & violet"]
        }

    def detect_card_type(self, text_content: str) -> PokemonCardType:
        """Detect the type of Pokémon card from OCR text."""
        text_lower = text_content.lower()
        
        # Check for specific type indicators
        if any(keyword in text_lower for keyword in self.type_keywords[PokemonCardType.POKEMON]):
            # Further classify Pokémon subtypes
            if "gx" in text_lower:
                return PokemonCardType.GX
            elif "ex" in text_lower and "gx" not in text_lower:
                return PokemonCardType.EX
            elif "vmax" in text_lower:
                return PokemonCardType.VMAX
            elif "vstar" in text_lower:
                return PokemonCardType.VSTAR
            elif " v " in text_lower or text_lower.endswith(" v"):
                return PokemonCardType.V
            elif "mega" in text_lower:
                return PokemonCardType.MEGA
            elif "radiant" in text_lower:
                return PokemonCardType.RADIANT
            elif "stage 2" in text_lower or "stage two" in text_lower:
                return PokemonCardType.STAGE_2
            elif "stage 1" in text_lower or "stage one" in text_lower:
                return PokemonCardType.STAGE_1
            else:
                return PokemonCardType.BASIC
                
        elif any(keyword in text_lower for keyword in self.type_keywords[PokemonCardType.TRAINER]):
            if "supporter" in text_lower:
                return PokemonCardType.SUPPORTER
            elif "item" in text_lower:
                return PokemonCardType.ITEM
            elif "stadium" in text_lower:
                return PokemonCardType.STADIUM
            elif "pokemon tool" in text_lower:
                return PokemonCardType.POKEMON_TOOL
            else:
                return PokemonCardType.TRAINER
                
        elif any(keyword in text_lower for keyword in self.type_keywords[PokemonCardType.ENERGY]):
            if "special" in text_lower:
                return PokemonCardType.SPECIAL_ENERGY
            else:
                return PokemonCardType.ENERGY
                
        return PokemonCardType.POKEMON  # Default assumption

    def detect_rarity(self, text_content: str, visual_features: Dict) -> PokemonRarity:
        """Detect card rarity from text and visual analysis."""
        text_lower = text_content.lower()
        
        # Check text indicators first
        for rarity, indicators in self.rarity_indicators.items():
            if any(indicator.lower() in text_lower for indicator in indicators):
                return rarity
        
        # Check visual features
        if visual_features.get("holographic_pattern", False):
            if visual_features.get("rainbow_effect", False):
                return PokemonRarity.RAINBOW_RARE
            elif visual_features.get("texture_detected", False):
                return PokemonRarity.TEXTURED
            else:
                return PokemonRarity.RARE_HOLO
        
        if visual_features.get("reverse_holo_pattern", False):
            return PokemonRarity.REVERSE_HOLO
            
        # Default based on card numbering if available
        card_number_match = re.search(r'(\d+)/(\d+)', text_content)
        if card_number_match:
            card_num, total_cards = map(int, card_number_match.groups())
            if card_num > total_cards:  # Secret rare numbering
                return PokemonRarity.SECRET_RARE
            elif card_num > total_cards * 0.85:  # Likely ultra rare
                return PokemonRarity.ULTRA_RARE
        
        return PokemonRarity.COMMON  # Default

    def detect_era(self, text_content: str, visual_features: Dict) -> PokemonCardEra:
        """Detect the era/generation of the Pokémon card."""
        text_lower = text_content.lower()
        
        # Check for era-specific indicators
        for era, indicators in self.era_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return era
        
        # Visual era detection
        if visual_features.get("yellow_border", False):
            return PokemonCardEra.VINTAGE
        elif visual_features.get("e_reader_strip", False):
            return PokemonCardEra.E_CARD
        elif visual_features.get("regulation_mark", False):
            return PokemonCardEra.SWORD_SHIELD  # Modern cards
            
        return PokemonCardEra.SWORD_SHIELD  # Default to modern

    def get_grading_criteria(self, card_type: PokemonCardType, rarity: PokemonRarity, 
                           era: PokemonCardEra) -> GradingCriteria:
        """Get specialized grading criteria for the specific card."""
        criteria = GradingCriteria(card_type, rarity, era)
        
        # Adjust weights based on card characteristics
        if rarity in [PokemonRarity.RARE_HOLO, PokemonRarity.ULTRA_RARE, 
                     PokemonRarity.SECRET_RARE, PokemonRarity.RAINBOW_RARE]:
            # Holographic cards judged more strictly on surface
            criteria.surface_weight = 30.0
            criteria.centering_weight = 30.0
            criteria.holographic_penalty_factor = 1.2
        
        if era == PokemonCardEra.VINTAGE:
            # Vintage cards have different tolerances
            criteria.vintage_tolerance = 1.1
            criteria.centering_weight = 40.0  # Centering very important for vintage
            
        if card_type in [PokemonCardType.GX, PokemonCardType.EX, 
                        PokemonCardType.V, PokemonCardType.VMAX]:
            # Modern ultra rares
            criteria.surface_weight = 28.0
            criteria.edges_weight = 22.0
            
        return criteria


@dataclass
class PokemonCardAnalysis:
    """Complete analysis results for a Pokémon card."""
    card_type: PokemonCardType
    rarity: PokemonRarity
    era: PokemonCardEra
    grading_criteria: GradingCriteria
    
    # Detected card information
    pokemon_name: Optional[str] = None
    set_name: Optional[str] = None
    card_number: Optional[str] = None
    
    # Visual characteristics
    has_holographic: bool = False
    has_texture: bool = False
    has_rainbow_effect: bool = False
    border_color: str = "silver"
    
    # Condition factors specific to Pokémon cards
    holo_scratches: int = 0
    edge_whitening: float = 0.0
    corner_wear: Dict[str, float] = None
    centering_measurements: Dict[str, float] = None
    
    # Overall assessment
    pokemon_specific_score: float = 0.0
    condition_notes: List[str] = None
    
    def __post_init__(self):
        if self.corner_wear is None:
            self.corner_wear = {"top_left": 0.0, "top_right": 0.0, 
                              "bottom_left": 0.0, "bottom_right": 0.0}
        if self.centering_measurements is None:
            self.centering_measurements = {"left": 0.0, "right": 0.0, 
                                         "top": 0.0, "bottom": 0.0}
        if self.condition_notes is None:
            self.condition_notes = []


# Pokémon-specific defect patterns
POKEMON_DEFECT_PATTERNS = {
    "holo_scratches": {
        "description": "Scratches visible on holographic foil",
        "severity_multiplier": 1.5,  # More severe on valuable holos
        "common_areas": ["center_artwork", "holo_pattern"]
    },
    "edge_whitening": {
        "description": "White showing on card edges from wear",
        "severity_multiplier": 1.2,
        "common_areas": ["all_edges"]
    },
    "corner_peeling": {
        "description": "Corner material separation or peeling",
        "severity_multiplier": 1.8,
        "common_areas": ["corners"]
    },
    "print_lines": {
        "description": "Horizontal lines from printing process",
        "severity_multiplier": 0.8,  # Often factory defects
        "common_areas": ["artwork", "text_areas"]
    },
    "centering_issues": {
        "description": "Card image not centered within borders",
        "severity_multiplier": 1.0,
        "common_areas": ["entire_card"]
    },
    "surface_indentations": {
        "description": "Dents or indentations in card surface",
        "severity_multiplier": 1.3,
        "common_areas": ["artwork", "text_box"]
    }
}


def get_pokemon_card_parts() -> PokemonCardParts:
    """Return the standard Pokémon card parts for analysis."""
    return PokemonCardParts()


def get_era_specific_considerations(era: PokemonCardEra) -> Dict[str, any]:
    """Get era-specific grading considerations."""
    considerations = {
        PokemonCardEra.VINTAGE: {
            "centering_tolerance": 0.05,  # More lenient
            "edge_wear_tolerance": 0.1,
            "print_quality_expectations": "lower",
            "common_defects": ["edge_whitening", "corner_wear", "centering"],
            "special_notes": "Factory cutting inconsistencies common"
        },
        PokemonCardEra.SCARLET_VIOLET: {
            "centering_tolerance": 0.02,  # Stricter
            "edge_wear_tolerance": 0.05,
            "print_quality_expectations": "high",
            "common_defects": ["print_lines", "surface_scratches"],
            "special_notes": "High print quality expected"
        }
    }
    
    # Default to modern for unlisted eras
    return considerations.get(era, considerations[PokemonCardEra.SWORD_SHIELD])