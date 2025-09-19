"""
Integration with official Pokémon TCG data from https://github.com/PokemonTCG/pokemon-tcg-data
Provides comprehensive card information for accurate grading and identification.
"""

import json
import os
import requests
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import re

from .card_types import PokemonCardType, PokemonRarity, PokemonCardEra

logger = logging.getLogger(__name__)


@dataclass
class TCGCardData:
    """Comprehensive card data from Pokémon TCG database."""
    # Basic identification
    id: str
    name: str
    supertype: str  # "Pokémon", "Trainer", "Energy"
    subtypes: List[str]
    
    # Set information
    set_id: str
    set_name: str
    set_series: str
    set_total: int
    number: str
    
    # Rarity and special characteristics
    rarity: str
    artist: Optional[str] = None
    
    # Pokémon-specific data
    hp: Optional[int] = None
    types: Optional[List[str]] = None
    evolves_from: Optional[str] = None
    evolves_to: Optional[List[str]] = None
    
    # Game mechanics
    attacks: Optional[List[Dict]] = None
    abilities: Optional[List[Dict]] = None
    weaknesses: Optional[List[Dict]] = None
    resistances: Optional[List[Dict]] = None
    retreat_cost: Optional[List[str]] = None
    
    # Pricing and market data
    tcgplayer_prices: Optional[Dict] = None
    cardmarket_prices: Optional[Dict] = None
    
    # Images
    images: Optional[Dict[str, str]] = None
    
    # Legality
    legalities: Optional[Dict[str, str]] = None
    
    # Set metadata
    release_date: Optional[str] = None
    updated_at: Optional[str] = None


class PokemonTCGDataManager:
    """Manager for Pokémon TCG data integration and card lookup."""
    
    def __init__(self, data_path: str = "data/pokemon-tcg-data"):
        self.data_path = data_path
        self.sets_data: Dict[str, Any] = {}
        self.cards_data: Dict[str, TCGCardData] = {}
        self.loaded = False
        
        # API endpoints for live data
        self.api_base = "https://api.pokemontcg.io/v2"
        self.github_base = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master"
        
        # Rarity mapping from TCG data to our internal system
        self.rarity_mapping = {
            "Common": PokemonRarity.COMMON,
            "Uncommon": PokemonRarity.UNCOMMON,
            "Rare": PokemonRarity.RARE,
            "Rare Holo": PokemonRarity.RARE_HOLO,
            "Rare Holo EX": PokemonRarity.ULTRA_RARE,
            "Rare Holo GX": PokemonRarity.ULTRA_RARE,
            "Rare Holo V": PokemonRarity.ULTRA_RARE,
            "Rare Holo VMAX": PokemonRarity.ULTRA_RARE,
            "Rare Secret": PokemonRarity.SECRET_RARE,
            "Rare Rainbow": PokemonRarity.RAINBOW_RARE,
            "Rare Ultra": PokemonRarity.ULTRA_RARE,
            "Amazing Rare": PokemonRarity.AMAZING_RARE,
            "Radiant Rare": PokemonRarity.RADIANT_RARE,
            "Classic Collection": PokemonRarity.ULTRA_RARE,
            "LEGEND": PokemonRarity.ULTRA_RARE,
            "Promo": PokemonRarity.PROMO
        }
        
        # Era mapping based on set series
        self.era_mapping = {
            "Base": PokemonCardEra.VINTAGE,
            "Jungle": PokemonCardEra.VINTAGE,
            "Fossil": PokemonCardEra.VINTAGE,
            "Team Rocket": PokemonCardEra.VINTAGE,
            "Neo": PokemonCardEra.VINTAGE,
            "Gym": PokemonCardEra.VINTAGE,
            "E-Card": PokemonCardEra.E_CARD,
            "EX": PokemonCardEra.EX,
            "Diamond & Pearl": PokemonCardEra.DIAMOND_PEARL,
            "Platinum": PokemonCardEra.DIAMOND_PEARL,
            "HeartGold & SoulSilver": PokemonCardEra.DIAMOND_PEARL,
            "Black & White": PokemonCardEra.BLACK_WHITE,
            "XY": PokemonCardEra.XY,
            "Sun & Moon": PokemonCardEra.SUN_MOON,
            "Sword & Shield": PokemonCardEra.SWORD_SHIELD,
            "Scarlet & Violet": PokemonCardEra.SCARLET_VIOLET
        }

    def load_data(self, force_download: bool = False) -> bool:
        """Load Pokémon TCG data from local files or download if needed."""
        try:
            if self.loaded and not force_download:
                return True
            
            # Check if local data exists
            if os.path.exists(self.data_path) and not force_download:
                self._load_local_data()
            else:
                # Download data from GitHub
                self._download_tcg_data()
                self._load_local_data()
            
            self.loaded = True
            logger.info(f"Loaded {len(self.cards_data)} cards from {len(self.sets_data)} sets")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TCG data: {e}")
            return False

    def _download_tcg_data(self):
        """Download Pokémon TCG data from GitHub repository."""
        logger.info("Downloading Pokémon TCG data from GitHub...")
        
        os.makedirs(self.data_path, exist_ok=True)
        
        # Download sets data
        sets_url = f"{self.github_base}/sets/en.json"
        response = requests.get(sets_url)
        if response.status_code == 200:
            with open(os.path.join(self.data_path, "sets.json"), "w") as f:
                json.dump(response.json(), f, indent=2)
        
        # Download cards data (this is large, so we'll start with a subset)
        # For demo purposes, download just the base sets
        base_sets = ["base1", "base2", "base3", "base4"]  # Base Set series
        
        for set_id in base_sets:
            try:
                cards_url = f"{self.github_base}/cards/en/{set_id}.json"
                response = requests.get(cards_url)
                if response.status_code == 200:
                    os.makedirs(os.path.join(self.data_path, "cards"), exist_ok=True)
                    with open(os.path.join(self.data_path, "cards", f"{set_id}.json"), "w") as f:
                        json.dump(response.json(), f, indent=2)
                    logger.info(f"Downloaded {set_id} card data")
            except Exception as e:
                logger.warning(f"Failed to download {set_id}: {e}")

    def _load_local_data(self):
        """Load data from local JSON files."""
        # Load sets data
        sets_file = os.path.join(self.data_path, "sets.json")
        if os.path.exists(sets_file):
            with open(sets_file, "r") as f:
                sets_data = json.load(f)
                for set_data in sets_data:
                    self.sets_data[set_data["id"]] = set_data
        
        # Load cards data
        cards_dir = os.path.join(self.data_path, "cards")
        if os.path.exists(cards_dir):
            for filename in os.listdir(cards_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(cards_dir, filename), "r") as f:
                        cards_data = json.load(f)
                        for card_data in cards_data:
                            tcg_card = self._parse_card_data(card_data)
                            self.cards_data[tcg_card.id] = tcg_card

    def _parse_card_data(self, card_data: Dict) -> TCGCardData:
        """Parse raw card data into TCGCardData object."""
        return TCGCardData(
            id=card_data.get("id", ""),
            name=card_data.get("name", ""),
            supertype=card_data.get("supertype", ""),
            subtypes=card_data.get("subtypes", []),
            set_id=card_data.get("set", {}).get("id", ""),
            set_name=card_data.get("set", {}).get("name", ""),
            set_series=card_data.get("set", {}).get("series", ""),
            set_total=card_data.get("set", {}).get("total", 0),
            number=card_data.get("number", ""),
            rarity=card_data.get("rarity", ""),
            artist=card_data.get("artist"),
            hp=card_data.get("hp"),
            types=card_data.get("types"),
            evolves_from=card_data.get("evolvesFrom"),
            evolves_to=card_data.get("evolvesTo"),
            attacks=card_data.get("attacks"),
            abilities=card_data.get("abilities"),
            weaknesses=card_data.get("weaknesses"),
            resistances=card_data.get("resistances"),
            retreat_cost=card_data.get("retreatCost"),
            tcgplayer_prices=card_data.get("tcgplayer", {}).get("prices"),
            cardmarket_prices=card_data.get("cardmarket", {}).get("prices"),
            images=card_data.get("images"),
            legalities=card_data.get("legalities"),
            release_date=card_data.get("set", {}).get("releaseDate"),
            updated_at=card_data.get("set", {}).get("updatedAt")
        )

    def identify_card(self, name: str, set_name: str = None, 
                     card_number: str = None) -> Optional[TCGCardData]:
        """Identify a card based on name and optional set information."""
        candidates = []
        
        # Search by exact name match first
        for card in self.cards_data.values():
            if card.name.lower() == name.lower():
                candidates.append(card)
        
        # If no exact match, try fuzzy matching
        if not candidates:
            name_lower = name.lower()
            for card in self.cards_data.values():
                if name_lower in card.name.lower() or card.name.lower() in name_lower:
                    candidates.append(card)
        
        if not candidates:
            return None
        
        # Filter by set name if provided
        if set_name:
            set_candidates = [c for c in candidates if set_name.lower() in c.set_name.lower()]
            if set_candidates:
                candidates = set_candidates
        
        # Filter by card number if provided
        if card_number:
            number_candidates = [c for c in candidates if c.number == card_number]
            if number_candidates:
                candidates = number_candidates
        
        # Return the first match (could be improved with better ranking)
        return candidates[0] if candidates else None

    def get_card_by_id(self, card_id: str) -> Optional[TCGCardData]:
        """Get card data by ID."""
        return self.cards_data.get(card_id)

    def search_cards(self, query: str, limit: int = 10) -> List[TCGCardData]:
        """Search for cards by name or other criteria."""
        results = []
        query_lower = query.lower()
        
        for card in self.cards_data.values():
            if (query_lower in card.name.lower() or 
                query_lower in card.set_name.lower() or
                query_lower in (card.artist or "").lower()):
                results.append(card)
                
                if len(results) >= limit:
                    break
        
        return results

    def get_rarity_from_tcg_data(self, tcg_rarity: str) -> PokemonRarity:
        """Convert TCG data rarity to internal rarity enum."""
        return self.rarity_mapping.get(tcg_rarity, PokemonRarity.COMMON)

    def get_era_from_tcg_data(self, set_series: str, release_date: str = None) -> PokemonCardEra:
        """Determine card era from set series and release date."""
        # Check series mapping first
        for series_key, era in self.era_mapping.items():
            if series_key.lower() in set_series.lower():
                return era
        
        # Fallback to date-based detection
        if release_date:
            try:
                year = int(release_date.split("-")[0])
                if year <= 2001:
                    return PokemonCardEra.VINTAGE
                elif year <= 2003:
                    return PokemonCardEra.E_CARD
                elif year <= 2007:
                    return PokemonCardEra.EX
                elif year <= 2011:
                    return PokemonCardEra.DIAMOND_PEARL
                elif year <= 2014:
                    return PokemonCardEra.BLACK_WHITE
                elif year <= 2017:
                    return PokemonCardEra.XY
                elif year <= 2020:
                    return PokemonCardEra.SUN_MOON
                elif year <= 2022:
                    return PokemonCardEra.SWORD_SHIELD
                else:
                    return PokemonCardEra.SCARLET_VIOLET
            except:
                pass
        
        return PokemonCardEra.SWORD_SHIELD  # Default to modern

    def get_card_type_from_tcg_data(self, supertype: str, subtypes: List[str]) -> PokemonCardType:
        """Convert TCG data types to internal card type enum."""
        if supertype == "Pokémon":
            # Check subtypes for specific classifications
            if "Basic" in subtypes:
                return PokemonCardType.BASIC
            elif "Stage 1" in subtypes:
                return PokemonCardType.STAGE_1
            elif "Stage 2" in subtypes:
                return PokemonCardType.STAGE_2
            elif "MEGA" in subtypes:
                return PokemonCardType.MEGA
            elif "EX" in subtypes:
                return PokemonCardType.EX
            elif "GX" in subtypes:
                return PokemonCardType.GX
            elif "V" in subtypes:
                return PokemonCardType.V
            elif "VMAX" in subtypes:
                return PokemonCardType.VMAX
            elif "VSTAR" in subtypes:
                return PokemonCardType.VSTAR
            elif "Radiant" in subtypes:
                return PokemonCardType.RADIANT
            else:
                return PokemonCardType.POKEMON
        
        elif supertype == "Trainer":
            if "Supporter" in subtypes:
                return PokemonCardType.SUPPORTER
            elif "Item" in subtypes:
                return PokemonCardType.ITEM
            elif "Stadium" in subtypes:
                return PokemonCardType.STADIUM
            elif "Pokémon Tool" in subtypes:
                return PokemonCardType.POKEMON_TOOL
            else:
                return PokemonCardType.TRAINER
        
        elif supertype == "Energy":
            if "Special" in subtypes:
                return PokemonCardType.SPECIAL_ENERGY
            else:
                return PokemonCardType.ENERGY
        
        return PokemonCardType.POKEMON  # Default

    def get_grading_adjustments(self, card_data: TCGCardData) -> Dict[str, float]:
        """Get grading adjustments based on card-specific factors."""
        adjustments = {
            "centering_weight": 35.0,
            "surface_weight": 25.0,
            "edges_weight": 20.0,
            "corners_weight": 20.0,
            "rarity_multiplier": 1.0,
            "era_tolerance": 1.0
        }
        
        # Adjust for rarity
        rarity = self.get_rarity_from_tcg_data(card_data.rarity)
        if rarity in [PokemonRarity.RARE_HOLO, PokemonRarity.ULTRA_RARE, 
                     PokemonRarity.SECRET_RARE]:
            adjustments["surface_weight"] = 30.0  # More focus on surface for holos
            adjustments["rarity_multiplier"] = 1.2
        
        # Adjust for era
        era = self.get_era_from_tcg_data(card_data.set_series, card_data.release_date)
        if era == PokemonCardEra.VINTAGE:
            adjustments["centering_weight"] = 40.0  # Centering crucial for vintage
            adjustments["era_tolerance"] = 1.1  # More lenient on minor defects
        
        # Adjust for special card types
        card_type = self.get_card_type_from_tcg_data(card_data.supertype, card_data.subtypes)
        if card_type in [PokemonCardType.EX, PokemonCardType.GX, PokemonCardType.V]:
            adjustments["surface_weight"] = 28.0
            adjustments["edges_weight"] = 22.0
        
        return adjustments

    def get_market_data(self, card_data: TCGCardData) -> Dict[str, Any]:
        """Extract market/pricing data for the card."""
        market_data = {
            "has_market_data": False,
            "tcgplayer_url": None,
            "average_price": None,
            "price_range": None,
            "last_updated": None
        }
        
        if card_data.tcgplayer_prices:
            market_data["has_market_data"] = True
            
            # Get price data (simplified - actual structure varies)
            if isinstance(card_data.tcgplayer_prices, dict):
                prices = []
                for condition, price_data in card_data.tcgplayer_prices.items():
                    if isinstance(price_data, dict) and "market" in price_data:
                        prices.append(price_data["market"])
                
                if prices:
                    market_data["average_price"] = sum(prices) / len(prices)
                    market_data["price_range"] = (min(prices), max(prices))
        
        return market_data

    def get_card_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded card database."""
        if not self.loaded:
            return {"error": "Data not loaded"}
        
        stats = {
            "total_cards": len(self.cards_data),
            "total_sets": len(self.sets_data),
            "cards_by_rarity": {},
            "cards_by_era": {},
            "cards_by_type": {}
        }
        
        for card in self.cards_data.values():
            # Count by rarity
            rarity = card.rarity
            stats["cards_by_rarity"][rarity] = stats["cards_by_rarity"].get(rarity, 0) + 1
            
            # Count by era
            era = self.get_era_from_tcg_data(card.set_series, card.release_date)
            era_name = era.value
            stats["cards_by_era"][era_name] = stats["cards_by_era"].get(era_name, 0) + 1
            
            # Count by type
            card_type = self.get_card_type_from_tcg_data(card.supertype, card.subtypes)
            type_name = card_type.value
            stats["cards_by_type"][type_name] = stats["cards_by_type"].get(type_name, 0) + 1
        
        return stats


# Global instance for use throughout the application
tcg_data_manager = PokemonTCGDataManager()


def initialize_tcg_data(force_download: bool = False) -> bool:
    """Initialize the TCG data manager."""
    return tcg_data_manager.load_data(force_download)


def get_card_info(name: str, set_name: str = None, card_number: str = None) -> Optional[TCGCardData]:
    """Get comprehensive card information."""
    if not tcg_data_manager.loaded:
        tcg_data_manager.load_data()
    
    return tcg_data_manager.identify_card(name, set_name, card_number)


def enhance_grading_with_tcg_data(card_name: str, detected_rarity: PokemonRarity,
                                 detected_era: PokemonCardEra) -> Dict[str, Any]:
    """Enhance grading analysis with official TCG data."""
    card_data = get_card_info(card_name)
    
    enhancement = {
        "tcg_data_found": card_data is not None,
        "official_rarity": None,
        "official_era": None,
        "grading_adjustments": {},
        "market_data": {},
        "card_details": {}
    }
    
    if card_data:
        enhancement["official_rarity"] = tcg_data_manager.get_rarity_from_tcg_data(card_data.rarity)
        enhancement["official_era"] = tcg_data_manager.get_era_from_tcg_data(
            card_data.set_series, card_data.release_date
        )
        enhancement["grading_adjustments"] = tcg_data_manager.get_grading_adjustments(card_data)
        enhancement["market_data"] = tcg_data_manager.get_market_data(card_data)
        enhancement["card_details"] = {
            "set_name": card_data.set_name,
            "card_number": card_data.number,
            "artist": card_data.artist,
            "release_date": card_data.release_date,
            "hp": card_data.hp,
            "types": card_data.types
        }
    
    return enhancement