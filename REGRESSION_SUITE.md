# Regressionssvit för chatboten

Detta dokument används för manuell regressionstestning efter deploy.

## Bedömningsmodell

Varje svar bedöms på fyra dimensioner:

1. Relevans
   - Besvarar svaret frågan direkt?
2. Täckning
   - Får användaren med de viktigaste punkterna?
3. Källstöd
   - Finns källor och verkar de rimliga för frågan?
4. Språk/klarhet
   - Är svaret begripligt, sammanhängande och fritt från brus?

Poäng per dimension:
- 0 = bristfälligt
- 1 = delvis godtagbart
- 2 = bra

Totalpoäng per fråga:
- 0-3 = svagt
- 4-6 = godtagbart
- 7-8 = bra

## Fördefinierade frågor

### Arbetsområden

Fråga: Vad är ett arbetsområde?
Förväntat fokus:
- Arbetsområde ska förklaras som en gruppering av aktiviteter
- Svaret bör nämna funktionell eller organisatorisk samhörighet
- Exempel bör vara arbetsområden som acceptanstest, utbildning eller IT-miljöer

Fråga: Hur används arbetsområden i planeringen?
Förväntat fokus:
- Arbetsområden ska kopplas till etapp- och aktivitetsplanering
- Svaret bör förklara att de används för struktur, ansvar och uppföljning

### Införandekrav

Fråga: Vad är syftet med införandekrav?
Förväntat fokus:
- Säkerställa att förutsättningar identifieras, hanteras och beslutas
- Omfatta verksamhet, teknik och organisation
- Resultat som minskad risk eller stabilare införande bör nämnas

Fråga: Vilka kravområden ingår i införandekrav?
Förväntat fokus:
- Svaret ska lista flera kravområden
- Områden som arbetsrutiner, systemsamband, utbildning, IT-miljöer och driftsättning bör finnas med

### Faser, etapper och aktiviteter

Fråga: Vilka huvudfaser ingår i införandet?
Förväntat fokus:
- Initiering
- Förberedelse
- Genomförande
- Avslut

Fråga: Vad är relationen mellan arbetsområden, etapper och aktiviteter?
Förväntat fokus:
- Arbetsområde = vad
- Etapp = när
- Aktivitet = hur

## Fria RAG-frågor byggda från PDF-underlaget

Fråga: Vilka kompetenser behövs för ett lyckat systeminförande?
Förväntat fokus:
- Systemkompetens
- Verksamhetskompetens
- Teknik- och driftkompetens
Primära källor:
- Checklista_Arbetsomraden.pdf
- Checklista_Inforandekrav.pdf

Fråga: Vad ska ingå i en aktivitetsbeskrivning?
Förväntat fokus:
- Syfte
- Ansvar
- Underlag
- Resultat
Primära källor:
- content.json
- Dokumentations- eller metodmaterial

Fråga: Hur används acceptanstest i införandet?
Förväntat fokus:
- Testplan
- Testfall
- Godkännandekriterier
- Verifiering i verksamheten
Primära källor:
- Checklista_Arbetsomraden.pdf
- Checklista_Inforandekrav.pdf
- Acceptanstest-planer

Fråga: Vilken roll har utbildning och användarstöd i införandet?
Förväntat fokus:
- Utbilda användare i system och arbetssätt
- Stödja lokala projekt och driftstart
- Kompletterande stödinsatser vid behov
Primära källor:
- Checklista_Arbetsomraden.pdf
- 253_Utbildning_slutrapport.pdf
- 240_Utbildning_strategi.pdf

Fråga: Hur bör systemsamband hanteras vid systeminförande?
Förväntat fokus:
- Beskriva kopplingar
- Testa och följa upp samband
- Beakta konsekvenser för omgivande system
Primära källor:
- Checklista_Arbetsomraden.pdf
- Checklista_Inforandekrav.pdf

Fråga: Vad behöver vara klart före driftsättning?
Förväntat fokus:
- Förberedelser
- Utbildning
- Konvertering
- Behörigheter
- Styrning och beslut
Primära källor:
- Checklista_Arbetsomraden.pdf
- Checklista_Inforandekrav.pdf
- Projektbeskrivningar

## Övergripande omdöme att sätta efter test

Bedöm chatboten som helhet på tre nivåer:

- Svag
  - Många svar är irrelevanta, tunna eller saknar rimligt källstöd
- Användbar
  - De flesta svar är relevanta och källorna rimliga, men resonemang eller precision brister ibland
- Bra
  - Svaren är konsekvent relevanta, väl förankrade i materialet och språkligt stabila
