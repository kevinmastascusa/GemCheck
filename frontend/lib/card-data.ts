import fs from 'fs';
import path from 'path';

interface PokemonCard {
  id: string;
  name: string;
  supertype: string;
  subtypes?: string[];
  hp?: string;
  types?: string[];
  evolvesFrom?: string;
  abilities?: Array<{
    name: string;
    text: string;
    type: string;
  }>;
  attacks?: Array<{
    name: string;
    cost?: string[];
    damage?: string;
    text?: string;
  }>;
  weaknesses?: Array<{
    type: string;
    value: string;
  }>;
  resistances?: Array<{
    type: string;
    value: string;
  }>;
  retreatCost?: string[];
  number: string;
  artist?: string;
  rarity?: string;
  flavorText?: string;
  nationalPokedexNumbers?: number[];
  images?: {
    small?: string;
    large?: string;
  };
  set?: {
    id: string;
    name: string;
    series?: string;
    printedTotal?: number;
    total?: number;
    legalities?: {
      unlimited?: string;
      standard?: string;
      expanded?: string;
    };
    ptcgoCode?: string;
    releaseDate?: string;
    updatedAt?: string;
    images?: {
      symbol?: string;
      logo?: string;
    };
  };
}

export class CardDataService {
  private static instance: CardDataService;
  private cardCache: Map<string, PokemonCard[]> = new Map();
  private allCards: PokemonCard[] = [];
  private isLoaded = false;

  static getInstance(): CardDataService {
    if (!CardDataService.instance) {
      CardDataService.instance = new CardDataService();
    }
    return CardDataService.instance;
  }

  async loadCardData(): Promise<void> {
    if (this.isLoaded) return;

    try {
      const dataPath = path.join(process.cwd(), '..', 'data', 'pokemon-tcg-data', 'cards', 'en');
      const files = fs.readdirSync(dataPath);
      
      // Load key sets first (most popular/recognizable cards)
      const prioritySets = ['base1.json', 'base2.json', 'base3.json', 'neo1.json', 'neo2.json'];
      const otherSets = files.filter(f => f.endsWith('.json') && !prioritySets.includes(f));
      
      const loadOrder = [...prioritySets, ...otherSets.slice(0, 20)]; // Load first 25 sets for performance
      
      for (const file of loadOrder) {
        try {
          const filePath = path.join(dataPath, file);
          const fileContent = fs.readFileSync(filePath, 'utf-8');
          const cards: PokemonCard[] = JSON.parse(fileContent);
          
          const setId = file.replace('.json', '');
          this.cardCache.set(setId, cards);
          this.allCards.push(...cards);
        } catch (error) {
          console.warn(`Failed to load card set ${file}:`, error);
        }
      }
      
      this.isLoaded = true;
      console.log(`Loaded ${this.allCards.length} cards from ${loadOrder.length} sets`);
    } catch (error) {
      console.error('Failed to load card data:', error);
      throw error;
    }
  }

  async searchCardsByName(name: string): Promise<PokemonCard[]> {
    await this.loadCardData();
    
    const searchTerm = name.toLowerCase().trim();
    return this.allCards.filter(card => 
      card.name.toLowerCase().includes(searchTerm)
    ).slice(0, 10); // Limit results
  }

  async searchCardsByNumber(setId: string, number: string): Promise<PokemonCard | null> {
    await this.loadCardData();
    
    const setCards = this.cardCache.get(setId);
    if (!setCards) return null;
    
    return setCards.find(card => card.number === number) || null;
  }

  async findCardByText(extractedText: {
    pokemonName?: string;
    setName?: string;
    cardNumber?: string;
    hp?: string;
    type?: string;
  }): Promise<PokemonCard[]> {
    await this.loadCardData();
    
    let candidates = this.allCards;
    
    // Filter by Pokemon name if provided
    if (extractedText.pokemonName) {
      const nameTerm = extractedText.pokemonName.toLowerCase();
      candidates = candidates.filter(card => 
        card.name.toLowerCase().includes(nameTerm)
      );
    }
    
    // Filter by HP if provided
    if (extractedText.hp) {
      candidates = candidates.filter(card => 
        card.hp === extractedText.hp
      );
    }
    
    // Filter by type if provided
    if (extractedText.type) {
      const typeTerm = extractedText.type.toLowerCase();
      candidates = candidates.filter(card => 
        card.types?.some(type => type.toLowerCase().includes(typeTerm))
      );
    }
    
    // Filter by card number if provided
    if (extractedText.cardNumber) {
      candidates = candidates.filter(card => 
        card.number === extractedText.cardNumber
      );
    }
    
    return candidates.slice(0, 5); // Return top 5 matches
  }

  async getRandomCard(): Promise<PokemonCard | null> {
    await this.loadCardData();
    
    if (this.allCards.length === 0) return null;
    
    // Prefer rare/holo cards for demo
    const rareCards = this.allCards.filter(card => 
      card.rarity?.includes('Rare') || card.rarity?.includes('Holo')
    );
    
    const pool = rareCards.length > 0 ? rareCards : this.allCards;
    const randomIndex = Math.floor(Math.random() * pool.length);
    return pool[randomIndex];
  }

  async getPopularCards(limit = 20): Promise<PokemonCard[]> {
    await this.loadCardData();
    
    // Get popular Pokemon from base sets
    const popularNames = [
      'Charizard', 'Blastoise', 'Venusaur', 'Pikachu', 'Mewtwo', 
      'Mew', 'Alakazam', 'Machamp', 'Gengar', 'Dragonite',
      'Lugia', 'Ho-Oh', 'Celebi', 'Rayquaza', 'Dialga', 'Palkia'
    ];
    
    const popularCards: PokemonCard[] = [];
    
    for (const name of popularNames) {
      const matches = await this.searchCardsByName(name);
      if (matches.length > 0) {
        // Prefer holo/rare versions
        const holo = matches.find(card => card.rarity?.includes('Holo'));
        popularCards.push(holo || matches[0]);
      }
      if (popularCards.length >= limit) break;
    }
    
    return popularCards;
  }

  getAllLoadedCards(): PokemonCard[] {
    return this.allCards;
  }
}