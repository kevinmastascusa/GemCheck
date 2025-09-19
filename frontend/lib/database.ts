import { prisma } from './prisma'
import type { PokemonCard, PSACard, CardAnalysis, User } from '@prisma/client'

export class DatabaseService {
  // Pokemon Card operations
  static async savePokemonCard(cardData: any) {
    try {
      return await prisma.pokemonCard.upsert({
        where: { id: cardData.id },
        update: {
          name: cardData.name,
          set: cardData.set?.name || cardData.set,
          setId: cardData.set?.id,
          number: cardData.number,
          rarity: cardData.rarity,
          artist: cardData.artist,
          flavorText: cardData.flavorText,
          hp: cardData.hp,
          types: cardData.types || [],
          supertype: cardData.supertype,
          subtypes: cardData.subtypes || [],
          nationalPokedexId: cardData.nationalPokedexNumbers?.[0],
          images: cardData.images,
          abilities: cardData.abilities,
          attacks: cardData.attacks,
          weaknesses: cardData.weaknesses,
          resistances: cardData.resistances,
          retreatCost: cardData.retreatCost || [],
          convertedRetreatCost: cardData.convertedRetreatCost,
        },
        create: {
          id: cardData.id,
          name: cardData.name,
          set: cardData.set?.name || cardData.set,
          setId: cardData.set?.id,
          number: cardData.number,
          rarity: cardData.rarity,
          artist: cardData.artist,
          flavorText: cardData.flavorText,
          hp: cardData.hp,
          types: cardData.types || [],
          supertype: cardData.supertype,
          subtypes: cardData.subtypes || [],
          nationalPokedexId: cardData.nationalPokedexNumbers?.[0],
          images: cardData.images,
          abilities: cardData.abilities,
          attacks: cardData.attacks,
          weaknesses: cardData.weaknesses,
          resistances: cardData.resistances,
          retreatCost: cardData.retreatCost || [],
          convertedRetreatCost: cardData.convertedRetreatCost,
        }
      })
    } catch (error) {
      console.error('Error saving Pokemon card:', error)
      throw error
    }
  }

  static async searchPokemonCards(query: string, limit = 20) {
    try {
      return await prisma.pokemonCard.findMany({
        where: {
          OR: [
            { name: { contains: query, mode: 'insensitive' } },
            { set: { contains: query, mode: 'insensitive' } },
          ]
        },
        take: limit,
        orderBy: { name: 'asc' }
      })
    } catch (error) {
      console.error('Error searching Pokemon cards:', error)
      throw error
    }
  }

  static async getPokemonCardById(id: string) {
    try {
      return await prisma.pokemonCard.findUnique({
        where: { id },
        include: {
          analyses: {
            take: 10,
            orderBy: { createdAt: 'desc' }
          },
          psaCards: true
        }
      })
    } catch (error) {
      console.error('Error getting Pokemon card:', error)
      throw error
    }
  }

  // PSA Card operations
  static async savePSACard(psaData: {
    certNumber: string
    grade: number
    cardName: string
    setName?: string
    year?: string
    cardNumber?: string
    imageUrl?: string
    psaUrl: string
    pokemonCardId?: string
    isVerified?: boolean
  }) {
    try {
      return await prisma.psaCard.upsert({
        where: { certNumber: psaData.certNumber },
        update: {
          grade: psaData.grade,
          cardName: psaData.cardName,
          setName: psaData.setName,
          year: psaData.year,
          cardNumber: psaData.cardNumber,
          imageUrl: psaData.imageUrl,
          psaUrl: psaData.psaUrl,
          pokemonCardId: psaData.pokemonCardId,
          isVerified: psaData.isVerified || false,
        },
        create: {
          certNumber: psaData.certNumber,
          grade: psaData.grade,
          cardName: psaData.cardName,
          setName: psaData.setName,
          year: psaData.year,
          cardNumber: psaData.cardNumber,
          imageUrl: psaData.imageUrl,
          psaUrl: psaData.psaUrl,
          pokemonCardId: psaData.pokemonCardId,
          isVerified: psaData.isVerified || false,
        }
      })
    } catch (error) {
      console.error('Error saving PSA card:', error)
      throw error
    }
  }

  static async getPSACardsByGrade(grade?: number, limit = 10) {
    try {
      const where = grade ? { grade } : {}
      return await prisma.psaCard.findMany({
        where,
        take: limit,
        orderBy: { createdAt: 'desc' },
        include: {
          pokemonCard: true
        }
      })
    } catch (error) {
      console.error('Error getting PSA cards by grade:', error)
      throw error
    }
  }

  static async verifyCertNumber(certNumber: string) {
    try {
      return await prisma.psaCard.findUnique({
        where: { certNumber },
        include: {
          pokemonCard: true
        }
      })
    } catch (error) {
      console.error('Error verifying cert number:', error)
      throw error
    }
  }

  // Card Analysis operations
  static async saveCardAnalysis(analysisData: {
    userId: string
    uploadId: string
    pokemonCardId: string
    predictedGrade: number
    confidence: number
    centeringScore: number
    edgesScore: number
    cornersScore: number
    surfaceScore: number
    overallScore: number
    defects?: any
    recognitionData?: any
    processingTime?: number
    analysisVersion?: string
  }) {
    try {
      return await prisma.cardAnalysis.create({
        data: analysisData,
        include: {
          pokemonCard: true,
          upload: true
        }
      })
    } catch (error) {
      console.error('Error saving card analysis:', error)
      throw error
    }
  }

  static async getAnalysesByUser(userId: string, limit = 50) {
    try {
      return await prisma.cardAnalysis.findMany({
        where: { userId },
        take: limit,
        orderBy: { createdAt: 'desc' },
        include: {
          pokemonCard: true,
          upload: true
        }
      })
    } catch (error) {
      console.error('Error getting user analyses:', error)
      throw error
    }
  }

  static async getAnalysisStats() {
    try {
      const [totalAnalyses, avgGrade, avgConfidence, gradeDistribution] = await Promise.all([
        prisma.cardAnalysis.count(),
        prisma.cardAnalysis.aggregate({
          _avg: { predictedGrade: true }
        }),
        prisma.cardAnalysis.aggregate({
          _avg: { confidence: true }
        }),
        prisma.cardAnalysis.groupBy({
          by: ['predictedGrade'],
          _count: { predictedGrade: true },
          orderBy: { predictedGrade: 'asc' }
        })
      ])

      return {
        totalAnalyses,
        averageGrade: avgGrade._avg.predictedGrade,
        averageConfidence: avgConfidence._avg.confidence,
        gradeDistribution
      }
    } catch (error) {
      console.error('Error getting analysis stats:', error)
      throw error
    }
  }

  // User operations
  static async createOrUpdateUser(userData: {
    email: string
    name?: string
  }) {
    try {
      return await prisma.user.upsert({
        where: { email: userData.email },
        update: { name: userData.name },
        create: userData
      })
    } catch (error) {
      console.error('Error creating/updating user:', error)
      throw error
    }
  }

  // System metrics
  static async updateSystemMetrics() {
    try {
      const [totalAnalyses, totalUsers, totalUploads, avgStats] = await Promise.all([
        prisma.cardAnalysis.count(),
        prisma.user.count(),
        prisma.cardUpload.count(),
        prisma.cardAnalysis.aggregate({
          _avg: {
            predictedGrade: true,
            confidence: true
          }
        })
      ])

      const popularCards = await prisma.cardAnalysis.groupBy({
        by: ['pokemonCardId'],
        _count: { pokemonCardId: true },
        orderBy: { _count: { pokemonCardId: 'desc' } },
        take: 10
      })

      return await prisma.systemMetrics.create({
        data: {
          totalAnalyses,
          totalUsers,
          totalUploads,
          averageGrade: avgStats._avg.predictedGrade,
          averageConfidence: avgStats._avg.confidence,
          popularCards: popularCards
        }
      })
    } catch (error) {
      console.error('Error updating system metrics:', error)
      throw error
    }
  }

  // Cleanup operations
  static async cleanupOldSessions() {
    try {
      const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000)
      return await prisma.analysisSession.updateMany({
        where: {
          startTime: { lt: oneDayAgo },
          isActive: true
        },
        data: {
          isActive: false,
          endTime: new Date()
        }
      })
    } catch (error) {
      console.error('Error cleaning up old sessions:', error)
      throw error
    }
  }
}