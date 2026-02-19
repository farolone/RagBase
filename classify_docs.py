"""Classify documents into collections based on title + author content."""
import json
import re
from rag.storage.postgres import PostgresStore

pg = PostgresStore()

# Get existing collections
colls_resp = pg.list_collections()
COLLECTIONS = {c["name"]: str(c["id"]) for c in colls_resp}
print("Collections:", list(COLLECTIONS.keys()))

# Keyword rules: collection_name -> (title_keywords, author_keywords)
# A doc matches if ANY title keyword OR author keyword matches (case-insensitive)
RULES = {
    "Auswandern": {
        "title": [r"auswander", r"wegzug", r"steueroase", r"offshore", r"expat",
                  r"perpetual travel", r"wohnsitz", r"steuerfreie?s?\b.*land",
                  r"freiheit.*sicherheit", r"flaggentheorie", r"non.?dom",
                  r"steuerlast", r"malta", r"dubai.*steuer", r"panama.*steuer",
                  r"paraguay", r"georgien.*steuer", r"besten.?länd",
                  r"raus aus.*1000", r"freiformation"],
        "author": [r"goodbye matrix", r"perspektive ausland", r"freiformation"],
    },
    "Bitcoin": {
        "title": [r"bitcoin", r"\bbtc\b", r"crypto", r"krypto", r"blockchain",
                  r"microstrategy", r"satoshi", r"saylor", r"altcoin", r"ethereum",
                  r"\beth\b", r"defi", r"hodl", r"halving", r"mining.*crypto",
                  r"ledger", r"wallet.*crypto", r"stablecoin",
                  r"blocktrainer"],
        "author": [r"blocktrainer"],
    },
    "Claude": {
        "title": [r"claude\b", r"anthropic", r"vibe.?cod", r"\bcursor\b",
                  r"windsurf", r"sub.?agent", r"cline", r"aider",
                  r"copilot.*cod", r"ai.*cod", r"cod.*ai\b",
                  r"clawdbot"],
        "author": [],
    },
    "Flying": {
        "title": [r"\bifr\b", r"\bvfr\b", r"pilot", r"flyi?n?g\b", r"aviat",
                  r"flight\b", r"mooney", r"\bvatsim\b", r"\bfsx\b", r"msfs",
                  r"\bppl\b", r"instrument.*approach", r"runway", r"\batc\b",
                  r"airspace", r"airplan[eé]", r"cessna", r"garmin.*g\d",
                  r"\bg1000\b", r"transition.*level", r"übergangs(höhe|schicht|flug)",
                  r"flugfläche", r"weather.*radar", r"metar", r"\btaf\b",
                  r"squawk", r"transponder", r"vor\b.*dme", r"\bnav\b.*log",
                  r"cross.?country", r"check.?ride", r"stall\b", r"approach.*plate",
                  r"holding.*pattern", r"RNAV", r"ILS\b", r"localizer",
                  r"glide.*slope", r"missed.*approach", r"flug(hafen|platz|zeug|funk|sicher)",
                  r"cirrus", r"piper", r"beech", r"bonanza",
                  r"single.*engine", r"multi.*engine", r"propeller",
                  r"kompass.*fehler", r"variometer", r"höhenmesser",
                  r"ground.*school", r"klappen", r"fahrwerk",
                  r"headwind", r"crosswind", r"tailwind",
                  r"density.*alt", r"icing", r"carb.*heat",
                  r"spin\b.*recover", r"aerobat", r"tomahawk",
                  r"aeroclub", r"ultraleicht", r"segelflu",
                  r"turboprop", r"drehfunkfeuer", r"cockpit.*takeoff",
                  r"carrier.*launch", r"cat.*launch",
                  r"diesel.*flugmotor", r"aircraft.*engine",
                  r"aircraft.*owner", r"compression.*test",
                  r"cylinder.*\d", r"lean.*peak", r"pre.?heat",
                  r"aborted.*takeoff", r"reciprocal.*heading",
                  r"wind.*correction.*angle", r"aircraft.*system",
                  r"sarajevo.*approach", r"transall", r"c.?160",
                  r"flugsicherung", r"f.?18.*cockpit"],
        "author": [r"aviation", r"pilot", r"bold method", r"fly8ma",
                   r"mooney", r"vatsim", r"flightchops", r"steveo1kinevo",
                   r"flight.*(insight|training|club)", r"guido",
                   r"captain.*joe", r"captain.*dan", r"ron\s*flying",
                   r"erau", r"specialvfr", r"blancolirio",
                   r"cheshunt", r"ty\s*manegold", r"aviana.*aircraft",
                   r"jay\s*kujan", r"mojogrip", r"cessna\s*flyer",
                   r"fliegermagazin", r"dfs\s*deutsche"],
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
                  r"alex.*krainer", r"patrik.*baab"],
        "author": [r"geopolit", r"quincy.*institute", r"open.*society",
                   r"glenn.*diesen", r"rfu.*news"],
    },
    "Health Management": {
        "title": [r"health", r"longevity", r"\bdiet\b", r"exercise",
                  r"fitness", r"sleep", r"aging", r"supplement",
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
                  r"kaffee.*einfluss", r"young.*aussehen"],
        "author": [r"huberman", r"attia", r"rhonda.*patrick",
                   r"dr\.?\s*berg", r"bob.*brad", r"athlean",
                   r"doctor.*myro", r"paul.*saladino",
                   r"doktorweigl", r"kolodenker"],
    },
    "Investing": {
        "title": [r"invest", r"portfolio", r"stock.*market", r"\betf\b",
                  r"dividend", r"\bfund\b", r"aktie[n]?", r"depot",
                  r"rendite", r"vermögen", r"geldanlage", r"s&?p.?500",
                  r"nasdaq", r"wall.*street", r"hedge\b", r"kalshi",
                  r"betting.*market", r"prediction.*market",
                  r"anleihe", r"bond\b", r"real.*estate",
                  r"immobilie", r"passive.*incom", r"kapital",
                  r"finanz", r"msci.*world", r"dax\b", r"börse",
                  r"market.*crash", r"recession", r"inflation",
                  r"federal.*reserve", r"zinsen", r"geld\b.*anlegen",
                  r"sparen", r"wealth", r"trader", r"trading",
                  r"optionen?\b", r"derivat", r"spread\b",
                  r"chartanalyse", r"technische.*analyse",
                  r"fundamental", r"bilanz", r"cashflow",
                  r"michael.*burry", r"nachkauf", r"nachrüst",
                  r"breakout.*2026", r"scanner.*stock",
                  r"asset.*invisible", r"make.*you.*poor",
                  r"bullish.*zyklus", r"green.*thumb",
                  r"larry.*fink", r"desaster.*europa.*trump",
                  r"industrie.*auslauf", r"wachstumsmo",
                  r"deutsche.*industrie.*herzkrank",
                  r"masse.*deutsch.*verarm",
                  r"enteignung.*gesetz",
                  r"debt.*explos"],
        "author": [r"mission.*money", r"finanzfluss", r"finanztip",
                   r"thorsten.*wittmann", r"andreas.*beck", r"gerd.*kommer",
                   r"hkcm", r"mario.*lochner", r"florian.*homm",
                   r"casgains", r"straight.*talk.*finance",
                   r"finanzmarktwelt", r"rené.*will.*rendite",
                   r"aktien.*mit.*kopf", r"bravos.*research",
                   r"treyding"],
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
                  r"proxmox", r"raspberry.*pi",
                  r"node.?red", r"\bknx\b", r"homematic",
                  r"scrapy", r"\bsdr\b.*radio",
                  r"docker\b", r"fastapi", r"neo4j", r"microservice",
                  r"large.*language.*model",
                  r"lidar.*blender", r"\bseo\b",
                  r"fingerprint.*demo", r"taws.*aws",
                  r"lightburn", r"pivot.*table",
                  r"starlink.*sirius"],
        "author": [r"matthew.*berman", r"jeremy.*howard", r"thu\s*vu",
                   r"david.*bombal", r"jeff.*geerling",
                   r"electronics.*wizard", r"github.*awesome",
                   r"haus.*automation", r"andreas.*spiess"],
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
                  r"marcell.*d.*avis", r"julian.*reichelt",
                  r"schwarz.?blau", r"tichy.*rede",
                  r"polizei.*unterschätz", r"silvester.*schlacht",
                  r"klima.*(studie|aktivi|schau|nachhilfe|rezo)",
                  r"wärmenachschlag", r"totaler.*stromausfall",
                  r"blackout", r"heat.*pump",
                  r"bürokratie.*habeck", r"sanktion.*wirtschaft",
                  r"kanzleramt.*klag", r"corona.*experten",
                  r"lanz.*scheint.*erwach",
                  r"gefügig.*angst.*verdumm",
                  r"wahnsinn.*zahle.*keinen.*cent",
                  r"waffen.*brücken.*schul",
                  r"ernst.*wolff", r"social.*engineer",
                  r"manipulation.*propaganda",
                  r"epstein.*dunkle",
                  r"not.*our.*audio",
                  r"menschen.*gefügig"],
        "author": [r"outdoor.*chiemgau", r"tichys.*einblick",
                   r"achtung.*reichelt", r"kettner",
                   r"paul.*brandenburg", r"compact",
                   r"hoss.*hopf", r"aya.*vel[áa]zquez",
                   r"bastian.*barucker", r"clownswelt",
                   r"flavio.*witzleben", r"snack.*cast",
                   r"mentale.*fitness", r"politik.*bär",
                   r"lacroix", r"amnas", r"will\s*vance",
                   r"michael\s*berlin.*360", r"glücksritter",
                   r"benjamin.*rothove",
                   r"hayek.*gesellschaft",
                   r"ketzer.*neuzeit", r"nordwolle"],
    },
}


def matches(text, patterns):
    """Check if text matches any regex pattern."""
    text_lower = text.lower()
    for p in patterns:
        if re.search(p, text_lower):
            return True
    return False


def classify(title, author):
    """Return list of matching collection names."""
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

    assigned_count = 0
    unassigned = []
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
            assigned_count += 1
        else:
            unassigned.append(f"{title[:65]} | {author}")

    print(f"{'='*55}")
    print(f"Assigned: {assigned_count}/{total}")
    print(f"Unassigned: {len(unassigned)}")
    print(f"{'='*55}")
    print(f"\nPer collection:")
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {count:>4} docs | {name}")

    if unassigned:
        print(f"\nUnassigned documents ({len(unassigned)}):")
        for u in unassigned:
            print(f"  - {u}")


if __name__ == "__main__":
    main()
