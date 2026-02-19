"""Classify documents into collections based on title + author content.
Creates 6 new collections and assigns all remaining unassigned docs."""
import json
import re
from rag.storage.postgres import PostgresStore

pg = PostgresStore()

# Create 6 new collections
NEW_COLLECTIONS = {
    "Musik": ("#EC4899", "Musik-Videos, Covers, Konzerte"),
    "DIY & Haustechnik": ("#F97316", "Heimwerken, Smart Home, Elektrik, Elektronik"),
    "Energie & Klima": ("#22C55E", "Energiewende, Wetter, Blackout, Batterien, Klima"),
    "Wissen & Bildung": ("#6366F1", "Lernen, Wissenschaft, Persönlichkeitsentwicklung"),
    "Auto & Mobilität": ("#EF4444", "Autos, Boote, Fahrzeuge, Motorsport"),
    "Unterhaltung": ("#A855F7", "YouTube Shorts, Comedy, Viral, Sonstiges"),
}

print("Creating new collections...")
for name, (color, desc) in NEW_COLLECTIONS.items():
    pg.create_collection(name, desc, color)
    print(f"  Created: {name}")

# Refresh collections from DB
colls_resp = pg.list_collections()
COLLECTIONS = {c["name"]: str(c["id"]) for c in colls_resp}
print(f"\nAll collections ({len(COLLECTIONS)}): {list(COLLECTIONS.keys())}")

# Get already-assigned doc IDs
assigned_ids = set()
for c in colls_resp:
    ids = pg.get_collection_document_ids(str(c["id"]))
    assigned_ids.update(ids)
print(f"Already assigned: {len(assigned_ids)} docs")

# Full keyword rules for ALL 15 collections
RULES = {
    "Auswandern": {
        "title": [r"auswander", r"wegzug", r"steueroase", r"offshore", r"expat",
                  r"perpetual travel", r"wohnsitz", r"steuerfreie?s?\b.*land",
                  r"freiheit.*sicherheit", r"flaggentheorie", r"non.?dom",
                  r"steuerlast", r"besten.?länd", r"überwachungsstaat.*vermögen"],
        "author": [r"goodbye matrix", r"perspektive ausland"],
    },
    "Bitcoin": {
        "title": [r"bitcoin", r"\bbtc\b", r"crypto", r"krypto", r"blockchain",
                  r"microstrategy", r"satoshi", r"saylor", r"altcoin", r"ethereum",
                  r"\beth\b", r"defi", r"hodl", r"halving", r"mining.*crypto",
                  r"ledger", r"wallet.*crypto", r"stablecoin",
                  r"blocktrainer", r"kryptowähr", r"nscc.*002.*bitcoin"],
        "author": [r"blocktrainer", r"simply\s*bitcoin", r"bitcoin\s*hotel",
                   r"altcoin\s*daily", r"kernergy"],
    },
    "Claude": {
        "title": [r"claude\b", r"anthropic", r"vibe.?cod", r"\bcursor\b",
                  r"windsurf", r"sub.?agent", r"cline", r"aider",
                  r"copilot.*cod", r"ai.*cod", r"cod.*ai\b",
                  r"clawdbot", r"switch.*accounts.*claude"],
        "author": [],
    },
    "Flying": {
        "title": [r"\bifr\b", r"\bvfr\b", r"pilot", r"flyi?n?g\b", r"aviat",
                  r"flight\b", r"mooney", r"\bvatsim\b", r"\bfsx\b", r"msfs",
                  r"\bppl\b", r"instrument.*approach", r"runway", r"\batc\b",
                  r"airspace", r"airplan[eé]", r"cessna", r"garmin.*g?\d",
                  r"\bg1000\b", r"transition.*level", r"übergangs(höhe|schicht|flug)",
                  r"flugfläche", r"weather.*radar", r"metar", r"\btaf\b",
                  r"squawk", r"transponder", r"vor\b.*dme",
                  r"cross.?country", r"check.?ride", r"approach.*plate",
                  r"holding.*pattern", r"RNAV", r"ILS\b", r"localizer",
                  r"glide.*slope", r"missed.*approach", r"flug(hafen|platz|zeug|funk|sicher)",
                  r"cirrus", r"piper", r"beech", r"bonanza",
                  r"single.*engine", r"multi.*engine", r"propeller",
                  r"kompass.*fehler", r"variometer", r"höhenmesser",
                  r"ground.*school", r"klappen", r"fahrwerk",
                  r"headwind", r"crosswind", r"tailwind",
                  r"density.*alt", r"icing\b", r"carb.*heat",
                  r"aerobat", r"tomahawk",
                  r"aeroclub", r"ultraleicht", r"segelflu",
                  r"turboprop", r"drehfunkfeuer", r"cockpit.*takeoff",
                  r"carrier.*launch", r"cat.*launch",
                  r"diesel.*flugmotor", r"aircraft.*engine",
                  r"aircraft.*owner", r"compression.*test",
                  r"cylinder.*\d", r"lean.*peak", r"pre.?heat",
                  r"aborted.*takeoff", r"reciprocal.*heading",
                  r"wind.*correction.*angle", r"aircraft.*system",
                  r"sarajevo.*approach", r"transall", r"c.?160",
                  r"flugsicherung", r"f.?18.*cockpit",
                  r"bush.*pilot", r"seaplane", r"caravan.*landing",
                  r"skagw", r"\bimc\b", r"instrument.*flying",
                  r"\bcfii\b", r"foreflight", r"lpv.*approach",
                  r"gfc.*500", r"autopilot.*overview",
                  r"garmin.*430", r"garmin.*530",
                  r"vectors.*final", r"clearance.*enroute",
                  r"pop.*up.*ifr", r"airworthiness.*directive",
                  r"g100ul", r"gami\b", r"sun.*fun",
                  r"icao.*flight.*plan", r"compulsory.*report",
                  r"lufträume.*deutschland", r"formation.*friend",
                  r"\bviperjet\b", r"stall.*slow.*motion",
                  r"anflug.*edq", r"anflug.*ed[a-z]{2}",
                  r"eddf.*tower", r"cleared.*visual",
                  r"fliegen.*italien", r"alpen.*flug",
                  r"single.*pilot.*ifr", r"savvy.*aviat",
                  r"go.?around.*mountain", r"paint.*touch.*up.*evelyn"],
        "author": [r"aviation", r"pilot", r"bold method", r"fly8ma",
                   r"mooney", r"vatsim", r"flightchops", r"steveo1kinevo",
                   r"flight.*(insight|training|club|instruction)",
                   r"captain.*joe", r"captain.*dan", r"ron\s*flying",
                   r"erau", r"specialvfr", r"blancolirio",
                   r"cheshunt", r"ty\s*manegold", r"aviana.*aircraft",
                   r"jay\s*kujan", r"mojogrip", r"cessna\s*flyer",
                   r"fliegermagazin", r"dfs\s*deutsche",
                   r"n228rm", r"sir\s*drifto", r"bushbird",
                   r"mike\s*vaccaro", r"pilotxtrim",
                   r"airmart", r"james\s*sloan", r"travis\s*knapp",
                   r"scott\s*hall", r"rmag", r"pilotsafety",
                   r"gian\s*luca", r"btflync", r"seth\s*lake",
                   r"loves2fly", r"free\s*pilot", r"finer\s*points",
                   r"michael\s*becher", r"guido\s*warnecke",
                   r"elmoradar", r"bryce\s*angell",
                   r"missionary.*bush.*pilot", r"alaska.*plane",
                   r"abbas.*aviation", r"genavi"],
    },
    "Geo politics": {
        "title": [r"geopolit", r"geostrateg", r"china.*war", r"russia.*war",
                  r"\bnato\b", r"ukraine", r"taiwan.*invas",
                  r"military.*strat", r"nuclear.*war", r"cold.*war",
                  r"world.*order", r"superpower", r"hegemony", r"empire\b",
                  r"world.*war.*3", r"ww3", r"\bwef\b", r"great.*reset",
                  r"kissinger", r"open.*society", r"karl.*popper",
                  r"new.*enem", r"foreign.*policy.*middle",
                  r"rand.*corporation", r"neoliberal.*debt",
                  r"economic.*apocalypse", r"us.?strateg",
                  r"nordstream", r"unterwerfung.*china",
                  r"bühne.*katastrophe", r"krieg.*gespräch",
                  r"canceln.*unterhaken", r"dekadenz.*westen",
                  r"nordkorea", r"tucker.*carlson.*putin",
                  r"alex.*krainer", r"patrik.*baab",
                  r"america.*collapse.*ww3",
                  r"palantir.*gaza", r"laboratory.*ai.*algo",
                  r"globalist.*reset"],
        "author": [r"geopolit", r"quincy.*institute", r"open.*society",
                   r"glenn.*diesen", r"rfu.*news", r"tom\s*bilyeu",
                   r"middle\s*east\s*eye", r"redacted"],
    },
    "Health Management": {
        "title": [r"health", r"longevity", r"\bdiet\b", r"exercise",
                  r"fitness", r"sleep\b", r"aging", r"supplement",
                  r"nutrition", r"fasting", r"biohack", r"testosterone",
                  r"hormone", r"vitami[ne]", r"protein", r"keto",
                  r"intermittent", r"cardio", r"strength.*train",
                  r"mental.*health", r"meditat", r"stress.*reduc",
                  r"immune", r"gut.*health", r"microbiom",
                  r"sauna", r"cold.*plung", r"cold.*shower",
                  r"zone.*2", r"vo2.*max", r"heart.*rate",
                  r"blood.*sugar", r"glucose", r"insulin",
                  r"inflammation", r"anti.?aging",
                  r"plantar.*fasci", r"hamstring.*stretch",
                  r"belly.*fat", r"burn.*fat", r"warts?\b",
                  r"skin.*tag", r"callu", r"cancer\b",
                  r"touch.*toes", r"pain.*relief", r"burnout",
                  r"kaffee.*einfluss", r"jung.*aussehen",
                  r"calcium.*plaque", r"arteries.*heart",
                  r"dark.*chocolate.*blood",
                  r"clean.*berries"],
        "author": [r"huberman", r"attia", r"rhonda.*patrick",
                   r"dr\.?\s*berg", r"bob.*brad", r"athlean",
                   r"doctor.*myro", r"paul.*saladino",
                   r"doktorweigl", r"kolodenker",
                   r"heart.*disease.*code", r"insulin.*resistant",
                   r"rangan.*chatterjee", r"bobby\s*parrish"],
    },
    "Investing": {
        "title": [r"invest", r"portfolio", r"stock.*market", r"\betf\b",
                  r"dividend", r"\bfund\b", r"aktie[n]?", r"depot",
                  r"rendite", r"vermögen", r"geldanlage", r"s&?p.?500",
                  r"nasdaq", r"wall.*street", r"hedge\b", r"kalshi",
                  r"betting.*market", r"prediction.*market",
                  r"anleihe", r"bond\b", r"real.*estate",
                  r"immobilie", r"passive.*incom", r"kapital",
                  r"finanz(?!ierung)", r"msci.*world", r"dax\b", r"börse",
                  r"market.*crash", r"recession", r"inflation\b",
                  r"federal.*reserve", r"zinsen", r"geld\b.*anlegen",
                  r"sparen", r"wealth", r"trader", r"trading",
                  r"optionen?\b", r"derivat", r"spread\b",
                  r"chartanalyse", r"technische.*analyse",
                  r"fundamental", r"bilanz", r"cashflow",
                  r"michael.*burry", r"nachkauf",
                  r"breakout.*2026", r"scanner.*stock",
                  r"asset.*invisible", r"make.*you.*poor",
                  r"bullish.*zyklus", r"green.*thumb",
                  r"larry.*fink", r"desaster.*europa.*trump",
                  r"industrie.*auslauf", r"wachstumsmo",
                  r"deutsche.*industrie.*herzkrank",
                  r"masse.*deutsch.*verarm",
                  r"enteignung.*gesetz",
                  r"debt.*explos",
                  r"rich.*money.*inflation",
                  r"anleger.*geld.*verlier",
                  r"financial.*system.*fail",
                  r"forecast.*asset.*price",
                  r"market.*algorithm",
                  r"market.*crash.*pattern",
                  r"gierig.*immobilie",
                  r"dirk.*müller.*zerstör",
                  r"inflationswelle.*vermögen",
                  r"schwache.*wirtschaft.*inflation",
                  r"bonus.*komm.*an"],
        "author": [r"mission.*money", r"finanzfluss", r"finanztip",
                   r"thorsten.*wittmann", r"andreas.*beck", r"gerd.*kommer",
                   r"hkcm", r"mario.*lochner", r"florian.*homm",
                   r"casgains", r"straight.*talk.*finance",
                   r"finanzmarktwelt", r"rené.*will.*rendite",
                   r"aktien.*mit.*kopf", r"bravos.*research",
                   r"treyding", r"stoic.*finance",
                   r"vermögensfabrik", r"dividenden.*backpacker",
                   r"proactive.*thinker", r"michael\s*cowan",
                   r"simon.*schoebel", r"investscience",
                   r"mark.*kohler", r"anish\b",
                   r"words.*rizdom", r"raoul.*pal",
                   r"biallo"],
    },
    "OpenClaw": {
        "title": [r"mcp\b", r"model.?context.?protocol", r"\bagent[s]?\b.*ai",
                  r"ai\s*agent", r"n8n", r"\brag\b", r"embed", r"vector",
                  r"knowledge.?graph", r"graphrag", r"fine.?tun",
                  r"hugging.?face", r"quantiz", r"\bllm\b", r"\bollama\b",
                  r"lm.?studio", r"deepseek", r"\bqwen\b", r"mistral",
                  r"\bgpt\b", r"openai", r"chatgpt", r"langchain",
                  r"llama\b", r"transformer", r"neural.*net",
                  r"machine.*learn", r"deep.*learn", r"tensor",
                  r"pytorch", r"dataset", r"benchmark",
                  r"inference", r"gpu.*train", r"token\b",
                  r"prompt.*engineer", r"retriev.*augment",
                  r"browser.*use.*agent", r"paperless.*ai",
                  r"automat.*document", r"smart.*home.*ai",
                  r"rerank", r"semantic.*search", r"ai.*autom",
                  r"self.*host.*ai", r"local.*ai", r"open.*source.*ai",
                  r"MLX\b", r"GGUF", r"lora\b",
                  r"google.*ai.*doomed", r"hacker.*guide.*language.*model",
                  r"local.*llm", r"finance.*local.*llm",
                  r"\bki\b.*tool", r"effizienter.*ki",
                  r"docker\b", r"fastapi", r"neo4j", r"microservice",
                  r"large.*language.*model", r"attention.*machine",
                  r"chatgpt.*terminal",
                  r"100.*automat.*agent",
                  r"did.*openai.*secretly.*gpt"],
        "author": [r"matthew.*berman", r"jeremy.*howard", r"thu\s*vu",
                   r"david.*bombal", r"github.*awesome",
                   r"ai.*automation", r"nick.*puru",
                   r"aicodek", r"fahd.*mirza",
                   r"steve.*builder", r"hundefined"],
    },
    "Politik": {
        "title": [r"\bpolitik\b", r"\brki\b", r"corona.*protokoll",
                  r"lockdown", r"impf(ung|pflicht|schad|stoff)", r"regierung",
                  r"bundestag", r"merkel", r"scholz", r"\bafd\b",
                  r"grüne[n]?\b", r"\bcdu\b", r"\bspd\b",
                  r"deutschland.*polit", r"verfassungs",
                  r"grundgesetz", r"meinungsfreiheit", r"zensur",
                  r"maßnahme[n]?.*corona", r"grundrecht",
                  r"migration.*deutsch", r"asyl", r"flüchtling",
                  r"energiewend", r"klimapolitik",
                  r"follow.*science.*rki", r"habeck", r"lauterbach",
                  r"böhmermann", r"reichelt", r"maaßen",
                  r"hoss.*hopf", r"windkraft.*potenzi",
                  r"ricarda.*lang", r"hubert.*aiwanger",
                  r"sendeverbot.*rt\b", r"stasi.*zersetz",
                  r"mainstream.*medien.*gewin", r"podcast.*zensier",
                  r"julian.*reichelt", r"schwarz.?blau",
                  r"tichy.*rede", r"polizei.*unterschätz",
                  r"silvester.*schlacht",
                  r"klima.*(studie|aktivi|schau|nachhilfe|rezo)",
                  r"totaler.*stromausfall", r"heat.*pump",
                  r"bürokratie.*habeck", r"sanktion.*wirtschaft",
                  r"kanzleramt.*klag", r"corona.*experten",
                  r"lanz.*scheint.*erwach",
                  r"gefügig.*angst.*verdumm",
                  r"wahnsinn.*zahle.*keinen.*cent",
                  r"waffen.*brücken.*schul",
                  r"ernst.*wolff", r"social.*engineer",
                  r"manipulation.*propaganda",
                  r"epstein.*dunkle",
                  r"menschen.*gefügig",
                  r"alice.*weidel", r"werteunion",
                  r"whistleblower.*rki",
                  r"eklat.*bundestag",
                  r"lüge.*regierung",
                  r"kanzler.*rede.*tichy",
                  r"china.*stahl.*deutsch",
                  r"entwicklungsland.*schuler",
                  r"klagen.*kanzleramt",
                  r"pv.*anlagen.*abgeschalt",
                  r"hätten.*kanzler",
                  r"ndr.*video.*stellung"],
        "author": [r"outdoor.*chiemgau", r"tichys.*einblick",
                   r"achtung.*reichelt", r"kettner",
                   r"paul.*brandenburg", r"compact",
                   r"hoss.*hopf", r"aya.*vel[áa]zquez",
                   r"bastian.*barucker", r"clownswelt",
                   r"flavio.*witzleben", r"snack.*cast",
                   r"mentale.*fitness", r"politik.*bär",
                   r"lacroix", r"amnas", r"will\s*vance",
                   r"michael\s*berlin.*360", r"glücksritter",
                   r"benjamin.*rothove", r"hayek.*gesellschaft",
                   r"ketzer.*neuzeit", r"nordwolle",
                   r"prof.*homburg", r"schuler.*fragen",
                   r"politics\s*check", r"für.*meinungsfreiheit",
                   r"landwirtschaft.*erleeben", r"anthony\s*lee"],
    },
    # === 6 NEW COLLECTIONS ===
    "Musik": {
        "title": [r"bohemian.*rhapsody", r"my\s*way\b.*remaster",
                  r"just.*the.*way.*you.*are", r"this.*is.*my.*life",
                  r"ain.*heavy.*brother", r"civil.*war.*gn.?r",
                  r"volare.*gipsy", r"pour.*some.*sugar",
                  r"favorite.*lick", r"hive.*festival.*hi.?tech",
                  r"chad.*smith.*nandi", r"magical.*moment.*piano",
                  r"diana.*krall", r"shirley.*bassey", r"frank.*sinatra"],
        "author": [r"diana\s*krall", r"shirley\s*bassey",
                   r"frank\s*sinatra", r"drum\s*channel",
                   r"jm\s*conciertos", r"romaric\s*gipsy",
                   r"maggie\s*baugh", r"aurélien.*froissart",
                   r"m\s*arantes"],
    },
    "DIY & Haustechnik": {
        "title": [r"lampenfassung", r"thermostatic.*cartridge",
                  r"homematic", r"smart.*meter", r"\bknx\b",
                  r"node.?red", r"bewässerungs",
                  r"bluetooth.*headset", r"gigaset",
                  r"schieflast", r"überspannung",
                  r"hot.*tub.*plastic.*cube",
                  r"usb.?c.*tutorial", r"connector.*cable.*pd",
                  r"apple.*notes.*iphone",
                  r"raspberry.*pi\b(?!.*sdr)", r"proxmox.*migrat",
                  r"lightburn\b", r"laser",
                  r"scrapy\b", r"web.*scraping",
                  r"pivot.*table", r"excel",
                  r"airpods.*replacement",
                  r"rhino.*cart", r"moving.*cart",
                  r"homeowner.*keeps.*happen",
                  r"bpmn.*model",
                  r"asmr.*volta",
                  r"how.*pair.*bluetooth"],
        "author": [r"proofwood", r"problemlöser",
                   r"haus.*automation", r"elv.*elektronik",
                   r"tech.*smart.*home", r"hager.*deutsch",
                   r"david.*handyman", r"cyfy\b",
                   r"speedy.*crafts", r"chandoo",
                   r"andreas\s*spiess", r"electronics.*wizard",
                   r"jeff\s*geerling", r"ligo"],
    },
    "Energie & Klima": {
        "title": [r"lifepo4", r"akku.*zellen", r"stromspeicher",
                  r"vanadium.*redox", r"flow.*battery",
                  r"sand.*battery", r"energiespeicher",
                  r"heat.*pump", r"wärmepumpe",
                  r"e.?fuel", r"brennstoffzelle", r"wasserstoff",
                  r"blackout\b", r"stromausfall",
                  r"schneesturm", r"wintereinbruch", r"frost",
                  r"hochsommer", r"wechselhaft", r"gewitter",
                  r"wärmenachschlag",
                  r"pv.*anlag", r"solaranlag", r"balkonkraftwerk",
                  r"energiewende.*konzept",
                  r"hans.?werner.*sinn.*energi",
                  r"globale.*energiewende",
                  r"imaginary.*thermometer",
                  r"power.*line.*balls"],
        "author": [r"kai\s*zorn", r"wetteronline",
                   r"solaranlage", r"schmid.*group",
                   r"tony\s*heller", r"cleo\s*abram",
                   r"zack.*films", r"petersberger",
                   r"grenzen.*wissens", r"4pi.*klima",
                   r"skill\s*builder"],
    },
    "Wissen & Bildung": {
        "title": [r"graduation.*master", r"diploma.*ceremony",
                  r"cognitive.*load", r"force.*brain.*study",
                  r"active.*recall", r"gpa\b",
                  r"change.*prozess.*lewin",
                  r"zuversicht.*gutes.*tun.*gabriel",
                  r"einstein.*marie.*curie",
                  r"wahre.*bedeutung.*git",
                  r"\bseo\b.*backlink",
                  r"customer.*complaint.*end"],
        "author": [r"yale.*school", r"ie\s*university",
                   r"sabine\s*hossenfelder", r"justin\s*sung",
                   r"darren\s*chai", r"organisationsentfalt",
                   r"bwstiftung", r"gbtec",
                   r"brain\s*boost", r"herr\s*programmierer",
                   r"nathan\s*gotch"],
    },
    "Auto & Mobilität": {
        "title": [r"bmw.*mercedes", r"s.?klasse",
                  r"tesla.*schrott", r"tesla.*neuwagen",
                  r"tesla.*pretend.*save",
                  r"reitzle.*ford.*premier",
                  r"catamaran.*skipper", r"sailing",
                  r"cheapest.*pick.?up.*truck",
                  r"vareta.*óleo.*moto",
                  r"tractor.*stunt"],
        "author": [r"alte\s*schule.*goldene.*ära",
                   r"danydrives", r"supercar\s*blondie",
                   r"omid\s*mouazzen", r"florence\s*anja",
                   r"anderson.*oliveira"],
    },
    "Unterhaltung": {
        "title": [r"#shorts\b", r"#short\b", r"#viral\b", r"#foryou",
                  r"#meme\b", r"#comedy\b", r"#respect\b",
                  r"would.*u.*do", r"luckiest.*video",
                  r"psychology.*facts.*women",
                  r"didn.*even.*hesitate",
                  r"dog.*fails.*soccer",
                  r"aperol.*spritz",
                  r"cooking.*pasta.*amazing",
                  r"giraffe.*shorts",
                  r"pigeon.*trap", r"bird.*trap",
                  r"boerboel.*meet",
                  r"disability.*logo.*painted",
                  r"frisbee.*golf.*shot",
                  r"dad.*joke.*akila",
                  r"amazing.*body.*girl",
                  r"damaged.*bags.*airline",
                  r"patrick.*swayze.*legendary",
                  r"mesmerising.*rock.*factory",
                  r"plastic.*bags.*banned",
                  r"woodstove.*outdoors",
                  r"ronaldo.*brüllkäfer",
                  r"common.*sense.*no.*longer",
                  r"snipers.*until",
                  r"chishazed.*coffeepod",
                  r"shaving.*callus",
                  r"sonalika.*tractor",
                  r"es.*wird.*wieder.*alles.*gut",
                  r"markus.*lanz.*500.*omr",
                  r"jokos.*app.*gescheitert",
                  r"Jeff.*Bezos.*yacht",
                  r"ripping.*hot.*too.*hot",
                  r"abonnieren.*jung.*aussehen",
                  r"deswegen.*männer.*frauen.*bezieh",
                  r"who.*would.*be.*down.*training",
                  r"reinhard.*flötotto.*amex",
                  r"ie.*graduation.*2022",
                  r"afr.*robo.*tech",
                  r"rechte.*gastes.*restaurant",
                  r"the.*audio.*fail",
                  r"relaxing.*waterfall.*sleep",
                  r"aviação.*agrícola"],
        "author": [r"rome\s*life\s*shorts", r"riley\s*mae",
                   r"yeahmad", r"casper\s*capital",
                   r"andy\s*the\s*sk", r"chisha\s*zed",
                   r"ajay\s*chakre", r"daily\s*facts",
                   r"sportsnation", r"viralhog",
                   r"paws_ng", r"aislan\s*mendes",
                   r"gadget\s*glimpse", r"leelu\s*gwala",
                   r"kagan\s*dunlap", r"wood\s*stove\s*jesse",
                   r"past\s*vision", r"inspired\s*mind",
                   r"the\s*stejfan", r"lisander",
                   r"armed\s*sphere", r"mohammed\s*ayan",
                   r"british\s*pathé", r"bob\s*reese",
                   r"shpetim", r"metroid",
                   r"the\s*respect\s*show",
                   r"immo\s*tommy", r"the\s*guy\b",
                   r"oezgoeren", r"céline.*willers",
                   r"gordon\s*ramsay", r"chris\s*young.*tested",
                   r"acquired\b", r"coast\s*pavement"],
    },
}


def matches(text, patterns):
    text_lower = text.lower()
    for p in patterns:
        if re.search(p, text_lower):
            return True
    return False


def classify(title, author):
    matched = []
    for coll_name, rules in RULES.items():
        if coll_name not in COLLECTIONS:
            continue
        if matches(title, rules["title"]) or matches(author, rules["author"]):
            matched.append(coll_name)
    return matched


def main():
    docs, total = pg.list_documents(limit=500)
    print(f"\n{total} documents to classify\n")

    new_assignments = 0
    still_unassigned = []
    stats = {name: 0 for name in COLLECTIONS}

    for doc in docs:
        doc_id = str(doc["id"])
        title = doc.get("title", "") or ""
        author = doc.get("author", "") or ""

        matched_colls = classify(title, author)

        if matched_colls:
            for coll_name in matched_colls:
                pg.add_document_to_collection(doc_id, COLLECTIONS[coll_name])
                stats[coll_name] += 1
            if doc_id not in assigned_ids:
                new_assignments += 1
        else:
            still_unassigned.append(f"{title[:65]} | {author}")

    print(f"{'='*55}")
    print(f"Previously assigned: {len(assigned_ids)}")
    print(f"Newly assigned:      {new_assignments}")
    print(f"Still unassigned:    {len(still_unassigned)}")
    print(f"{'='*55}")
    print(f"\nPer collection:")
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {count:>4} docs | {name}")

    if still_unassigned:
        print(f"\nStill unassigned ({len(still_unassigned)}):")
        for u in still_unassigned:
            print(f"  - {u}")


if __name__ == "__main__":
    main()
