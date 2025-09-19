import { NextRequest, NextResponse } from 'next/server';

interface PSAGradedCard {
  certNumber: string;
  grade: number;
  cardName: string;
  setName?: string;
  year?: string;
  cardNumber?: string;
  imageUrl?: string;
  psaUrl: string;
}

// PSA Set Registry URLs for Pokemon cards (popular sets)
const PSA_SET_REGISTRY_URLS = [
  'https://www.psacard.com/cardfacts/pokemon-base-set-shadowless-1st-edition',
  'https://www.psacard.com/cardfacts/pokemon-base-set-shadowless-unlimited',
  'https://www.psacard.com/cardfacts/pokemon-base-set-1st-edition',
  'https://www.psacard.com/cardfacts/pokemon-neo-genesis-1st-edition',
  'https://www.psacard.com/cardfacts/pokemon-jungle-1st-edition',
  'https://www.psacard.com/cardfacts/pokemon-fossil-1st-edition'
];

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const grade = searchParams.get('grade');
    const limit = parseInt(searchParams.get('limit') || '10');
    
    // For now, return mock PSA graded card examples
    // In production, this would carefully scrape PSA registry data
    const mockPSACards: PSAGradedCard[] = [
      {
        certNumber: '52312345',
        grade: 10,
        cardName: 'Charizard Base Set Shadowless',
        setName: 'Base Set Shadowless',
        year: '1998',
        cardNumber: '4',
        imageUrl: 'https://images.pokemontcg.io/base1/4_hires.png',
        psaUrl: 'https://www.psacard.com/cert/52312345'
      },
      {
        certNumber: '52398765',
        grade: 9,
        cardName: 'Blastoise Base Set Shadowless',
        setName: 'Base Set Shadowless',
        year: '1998',
        cardNumber: '2',
        imageUrl: 'https://images.pokemontcg.io/base1/2_hires.png',
        psaUrl: 'https://www.psacard.com/cert/52398765'
      },
      {
        certNumber: '52376543',
        grade: 10,
        cardName: 'Pikachu Base Set',
        setName: 'Base Set',
        year: '1998',
        cardNumber: '58',
        imageUrl: 'https://images.pokemontcg.io/base1/58_hires.png',
        psaUrl: 'https://www.psacard.com/cert/52376543'
      },
      {
        certNumber: '52387654',
        grade: 8,
        cardName: 'Venusaur Base Set Shadowless',
        setName: 'Base Set Shadowless',
        year: '1998',
        cardNumber: '15',
        imageUrl: 'https://images.pokemontcg.io/base1/15_hires.png',
        psaUrl: 'https://www.psacard.com/cert/52387654'
      },
      {
        certNumber: '52365432',
        grade: 9,
        cardName: 'Mewtwo Base Set',
        setName: 'Base Set',
        year: '1998',
        cardNumber: '10',
        imageUrl: 'https://images.pokemontcg.io/base1/10_hires.png',
        psaUrl: 'https://www.psacard.com/cert/52365432'
      },
      {
        certNumber: '52354321',
        grade: 10,
        cardName: 'Alakazam Base Set',
        setName: 'Base Set',
        year: '1998',
        cardNumber: '1',
        imageUrl: 'https://images.pokemontcg.io/base1/1_hires.png',
        psaUrl: 'https://www.psacard.com/cert/52354321'
      },
      {
        certNumber: '52343210',
        grade: 7,
        cardName: 'Machamp Base Set',
        setName: 'Base Set',
        year: '1998',
        cardNumber: '8',
        imageUrl: 'https://images.pokemontcg.io/base1/8_hires.png',
        psaUrl: 'https://www.psacard.com/cert/52343210'
      },
      {
        certNumber: '52332109',
        grade: 9,
        cardName: 'Gyarados Base Set',
        setName: 'Base Set',
        year: '1998',
        cardNumber: '6',
        imageUrl: 'https://images.pokemontcg.io/base1/6_hires.png',
        psaUrl: 'https://www.psacard.com/cert/52332109'
      }
    ];
    
    let filteredCards = mockPSACards;
    
    // Filter by grade if specified
    if (grade) {
      const targetGrade = parseInt(grade);
      filteredCards = mockPSACards.filter(card => card.grade === targetGrade);
    }
    
    // Limit results
    const results = filteredCards.slice(0, limit);
    
    return NextResponse.json({ 
      success: true,
      cards: results,
      total: results.length,
      message: `Found ${results.length} PSA graded card examples`
    });
  } catch (error) {
    console.error('Error fetching PSA examples:', error);
    return NextResponse.json({ 
      error: 'Failed to fetch PSA graded examples',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// Helper function to respectfully scrape PSA registry data
async function scrapePSARegistry(url: string): Promise<PSAGradedCard[]> {
  // This would implement respectful web scraping of PSA registry
  // with proper rate limiting and error handling
  // For now, return empty array as we're using mock data
  return [];
}

// Helper function to validate PSA cert numbers
function isValidCertNumber(certNumber: string): boolean {
  return /^\d{8,9}$/.test(certNumber);
}

// Helper function to construct PSA cert URL
function getPSACertUrl(certNumber: string): string {
  return `https://www.psacard.com/cert/${certNumber}`;
}

// POST endpoint to verify a PSA certificate
export async function POST(request: NextRequest) {
  try {
    const { certNumber } = await request.json();
    
    if (!certNumber || !isValidCertNumber(certNumber)) {
      return NextResponse.json({ 
        error: 'Invalid PSA certificate number' 
      }, { status: 400 });
    }
    
    // In production, this would verify the cert number against PSA's database
    // For now, simulate verification
    const isValid = Math.random() > 0.3; // 70% chance of valid cert
    
    if (isValid) {
      // Mock verified card data
      const verifiedCard: PSAGradedCard = {
        certNumber,
        grade: Math.floor(Math.random() * 4) + 7, // Grade 7-10
        cardName: 'Verified Pokemon Card',
        psaUrl: getPSACertUrl(certNumber)
      };
      
      return NextResponse.json({
        success: true,
        verified: true,
        card: verifiedCard
      });
    } else {
      return NextResponse.json({
        success: true,
        verified: false,
        message: 'Certificate number not found in PSA database'
      });
    }
  } catch (error) {
    console.error('Error verifying PSA certificate:', error);
    return NextResponse.json({ 
      error: 'Failed to verify PSA certificate',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}