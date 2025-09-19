import { NextRequest, NextResponse } from 'next/server';

interface PokemonCardImage {
  id: string;
  name: string;
  imageUrl: string;
  set: string;
  number: string;
}

// Pokemon TCG API endpoint for high-quality card images
const POKEMON_TCG_API = 'https://api.pokemontcg.io/v2/cards';

// Popular Pokemon cards for demonstration
const POPULAR_POKEMON_CARDS = [
  { name: 'Charizard', set: 'base1', number: '4' },
  { name: 'Blastoise', set: 'base1', number: '2' },
  { name: 'Venusaur', set: 'base1', number: '15' },
  { name: 'Pikachu', set: 'base1', number: '58' },
  { name: 'Mewtwo', set: 'base1', number: '10' },
  { name: 'Mew', set: 'base1', number: '8' },
  { name: 'Alakazam', set: 'base1', number: '1' },
  { name: 'Machamp', set: 'base1', number: '8' },
];

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const query = searchParams.get('q') || '';
    const limit = parseInt(searchParams.get('limit') || '20');
    
    // If no query, return popular cards
    if (!query) {
      const randomCards = POPULAR_POKEMON_CARDS
        .sort(() => 0.5 - Math.random())
        .slice(0, limit);
      
      const cardPromises = randomCards.map(async (card) => {
        try {
          const response = await fetch(
            `${POKEMON_TCG_API}?q=name:${card.name} set.id:${card.set}&pageSize=1`,
            {
              headers: {
                'X-Api-Key': process.env.POKEMON_TCG_API_KEY || '',
              },
            }
          );
          
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const data = await response.json();
          
          if (data.data && data.data.length > 0) {
            const pokemonCard = data.data[0];
            return {
              id: pokemonCard.id,
              name: pokemonCard.name,
              imageUrl: pokemonCard.images.large,
              set: pokemonCard.set.name,
              number: pokemonCard.number,
            };
          }
        } catch (error) {
          console.error(`Error fetching ${card.name}:`, error);
          return null;
        }
      });
      
      const results = await Promise.all(cardPromises);
      const validResults = results.filter((card): card is PokemonCardImage => card !== null);
      
      return NextResponse.json({ cards: validResults });
    }
    
    // Search for specific Pokemon cards
    const response = await fetch(
      `${POKEMON_TCG_API}?q=name:${encodeURIComponent(query)}*&pageSize=${limit}`,
      {
        headers: {
          'X-Api-Key': process.env.POKEMON_TCG_API_KEY || '',
        },
      }
    );
    
    if (!response.ok) {
      throw new Error(`Pokemon TCG API error: ${response.status}`);
    }
    
    const data = await response.json();
    
    const cards: PokemonCardImage[] = data.data.map((card: any) => ({
      id: card.id,
      name: card.name,
      imageUrl: card.images.large,
      set: card.set.name,
      number: card.number,
    }));
    
    return NextResponse.json({ cards });
  } catch (error) {
    console.error('Error fetching Pokemon cards:', error);
    
    // Fallback to demo data if API fails
    const fallbackCards: PokemonCardImage[] = [
      {
        id: 'demo-1',
        name: 'Charizard',
        imageUrl: 'https://images.pokemontcg.io/base1/4_hires.png',
        set: 'Base Set',
        number: '4',
      },
      {
        id: 'demo-2',
        name: 'Blastoise',
        imageUrl: 'https://images.pokemontcg.io/base1/2_hires.png',
        set: 'Base Set',
        number: '2',
      },
      {
        id: 'demo-3',
        name: 'Venusaur',
        imageUrl: 'https://images.pokemontcg.io/base1/15_hires.png',
        set: 'Base Set',
        number: '15',
      },
    ];
    
    return NextResponse.json({ cards: fallbackCards });
  }
}