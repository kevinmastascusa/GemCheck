# PostgreSQL Setup for PSA Pre-Grader

## Manual Installation (Recommended)

Since automated installation requires admin privileges, please install PostgreSQL manually:

### Option 1: Download from Official Site
1. Go to https://www.postgresql.org/download/windows/
2. Download PostgreSQL 17 (or latest version)
3. Run the installer as administrator
4. During installation:
   - Set password for `postgres` user (remember this!)
   - Default port: 5432
   - Default locale: English, United States

### Option 2: Using Chocolatey (if available)
Run PowerShell as Administrator and execute:
```powershell
choco install postgresql17 -y
```

### Option 3: Using winget (alternative)
Run Command Prompt as Administrator:
```cmd
winget install PostgreSQL.PostgreSQL.17
```

## After Installation

1. **Start PostgreSQL Service**
   - Press `Win + R`, type `services.msc`
   - Find "postgresql-x64-17" service
   - Right-click → Start (if not already running)

2. **Create Database**
   Open Command Prompt and run:
   ```cmd
   createdb -U postgres psa_pregrader
   ```

3. **Set Environment Variables**
   Copy `.env.example` to `.env` in the frontend folder:
   ```bash
   cp .env.example .env
   ```
   
   Update the DATABASE_URL in `.env`:
   ```
   DATABASE_URL="postgresql://postgres:your_password@localhost:5432/psa_pregrader?schema=public"
   ```
   Replace `your_password` with the password you set during installation.

4. **Run Database Migration**
   ```bash
   cd frontend
   npx prisma migrate dev --name init
   ```

5. **Seed the Database (Optional)**
   ```bash
   npx prisma db seed
   ```

## Verification

Test the connection:
```bash
cd frontend
npx prisma studio
```
This should open Prisma Studio in your browser at http://localhost:5555

## Troubleshooting

### Connection Issues
- Ensure PostgreSQL service is running
- Check firewall settings
- Verify username/password in DATABASE_URL

### Permission Issues
- Make sure the `postgres` user has proper permissions
- Try connecting with pgAdmin if installed

### Port Conflicts
- Default port 5432 might be in use
- Change port in postgresql.conf and update DATABASE_URL

## Next Steps

After PostgreSQL is set up:
1. The database schema is already defined in `prisma/schema.prisma`
2. Run migrations to create tables: `npx prisma migrate dev`
3. Generate Prisma client: `npx prisma generate`
4. Start using the database in your PSA Pre-Grader application!

## Database Schema Overview

The PSA Pre-Grader database includes:
- **Users**: User accounts and authentication
- **PokemonCard**: Pokemon card data from TCG API
- **PSACard**: PSA graded card examples and certificates
- **CardUpload**: File upload tracking
- **CardAnalysis**: Analysis results and grading predictions
- **AnalysisSession**: User session tracking
- **SystemMetrics**: Application analytics
- **CardTemplate**: Template data for card recognition