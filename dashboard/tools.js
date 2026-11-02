// Shared by the Scam Checker (deal.html) and the Smart Basket (basket.html).
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const inr = v => v == null ? "–" : "₹" + Math.round(v).toLocaleString("en-IN");

// Store names as buyers know them ("Amazon.in" and "amazon.in" are one store).
const PLATFORMS = [["amazon", "Amazon"], ["shopsy", "Shopsy (Flipkart)"], ["flipkart", "Flipkart"], ["jiomart", "JioMart"], ["croma", "Croma"],
  ["reliance", "Reliance Digital"], ["tata cliq", "Tata CLiQ"], ["tatacliq", "Tata CLiQ"], ["vijay", "Vijay Sales"], ["myntra", "Myntra"], ["ajio", "AJIO"], ["nykaa", "Nykaa"],
  ["blinkit", "Blinkit"], ["zepto", "Zepto"], ["bigbasket", "BigBasket"], ["instamart", "Instamart"], ["meesho", "Meesho"], ["dmart", "DMart"], ["snapdeal", "Snapdeal"],
  ["apollo", "Apollo"], ["pharmeasy", "PharmEasy"], ["1mg", "1mg"], ["purplle", "Purplle"], ["firstcry", "FirstCry"], ["decathlon", "Decathlon"]];
const platform = s => { const t = (s || "").toLowerCase(); const hit = PLATFORMS.find(([k]) => t.includes(k)); return hit ? hit[1] : (s || "").trim(); };

// Server status chip, same as the search page.
async function connect() {
  let h = null;
  try { const r = await fetch("/api/health"); h = r.ok ? await r.json() : null; } catch {}
  const st = $("#status"); st.classList.add(h && h.search ? "live" : "demo");
  st.lastElementChild.textContent = h && h.search ? "Live search connected" : "Server not connected";
  return !!(h && h.search);
}

async function liveSearch(q) {
  const fd = new FormData(); fd.append("q", q);
  const r = await fetch("/api/search", { method: "POST", body: fd });
  const data = await r.json(); if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

// The listings the search page would trust for the searched product: exact product, trusted store (big store,
// authorised reseller, brand store), new, not an accessory, photo not flagged, and not suspiciously far below the
// fair price (a store selling its own stock may be up to 60% below, the same rule as the search page).
function trustedOffers(D) {
  const O = (D.offers || []).filter(o => !o.is_ref && o.match === "exact");
  // ...unless the search names a size ("200g") and the listing does not state one (probably a smaller pack)
  const typ = D.typical, sizeAsked = !!(D.query_attrs || {}).size;
  return O.filter(o => o.known && o.price_value != null && (o.condition || "new") === "new" && !o.accessory && !o.photo_mismatch
    && !(typ && o.price_value < 0.6 * typ && !(o.first_party && o.price_value >= 0.4 * typ && (!sizeAsked || (o.attrs || {}).size))));
}

// ---------- fake store websites: lookalike spellings, brand names on other domains, risky endings ----------
const OFFICIAL = {
  amazon: ["amazon.in", "amazon.com"], flipkart: ["flipkart.com"], myntra: ["myntra.com"], ajio: ["ajio.com"], croma: ["croma.com"],
  reliancedigital: ["reliancedigital.in"], jiomart: ["jiomart.com"], meesho: ["meesho.com"], nykaa: ["nykaa.com"], tatacliq: ["tatacliq.com"],
  vijaysales: ["vijaysales.com"], bigbasket: ["bigbasket.com"], blinkit: ["blinkit.com"], zepto: ["zeptonow.com", "zepto.com"], snapdeal: ["snapdeal.com"],
  apple: ["apple.com"], samsung: ["samsung.com"], oneplus: ["oneplus.in", "oneplus.com"], boat: ["boat-lifestyle.com"], decathlon: ["decathlon.in"],
  firstcry: ["firstcry.com"], purplle: ["purplle.com"], lenskart: ["lenskart.com"], swiggy: ["swiggy.com"], shopsy: ["shopsy.in"],
};
const NAMES = { amazon: "Amazon", flipkart: "Flipkart", myntra: "Myntra", ajio: "AJIO", croma: "Croma", reliancedigital: "Reliance Digital", jiomart: "JioMart",
  meesho: "Meesho", nykaa: "Nykaa", tatacliq: "Tata CLiQ", vijaysales: "Vijay Sales", bigbasket: "BigBasket", blinkit: "Blinkit", zepto: "Zepto", snapdeal: "Snapdeal",
  apple: "Apple", samsung: "Samsung", oneplus: "OnePlus", boat: "boAt", decathlon: "Decathlon", firstcry: "FirstCry", purplle: "Purplle", lenskart: "Lenskart",
  swiggy: "Swiggy", shopsy: "Shopsy" };
const RISKY_ENDINGS = ["shop", "xyz", "top", "store", "online", "site", "live", "buzz", "click", "club", "icu", "vip", "sale", "deals", "cfd", "sbs", "cyou", "rest", "fun", "today"];
const SOCIAL = /instagram|insta\b|whatsapp|telegram|facebook|fb\b|marketplace|dm\b|olx|quikr/i;

function levenshtein(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++)
    d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}
// letters scammers swap in: amaz0n, f1ipkart, rn for m
const unswap = s => s.replace(/rn/g, "m").replace(/vv/g, "w").replace(/0/g, "o").replace(/1/g, "l").replace(/3/g, "e").replace(/4/g, "a").replace(/5/g, "s").replace(/7/g, "t").replace(/[-_.]/g, "");

// -> {kind: "official" | "fake" | "risky" | "unknown" | "social" | "store-name" | "none", store?, reasons: [[level, text]]}
function checkSeller(input) {
  const raw = (input || "").trim();
  if (!raw) return { kind: "none", reasons: [] };
  const reasons = [];
  if (SOCIAL.test(raw)) return { kind: "social", reasons: [["bad", "Sold through social media or a classifieds app: no buyer protection, no returns, and payment usually goes straight to a person."]] };
  const looksUrl = /[./]/.test(raw) && !/\s/.test(raw);
  if (!looksUrl) {   // a store name, not a website
    const p = platform(raw), known = PLATFORMS.some(([k]) => raw.toLowerCase().includes(k));
    return known ? { kind: "store-name", store: p, reasons: [["good", `${p} is a well-known store. Make sure the link you use really is ${p}'s official website or app.`]] }
                 : { kind: "unknown", reasons: [["warn", `"${raw}" is not a store DupeLens knows.`]] };
  }
  let host = raw.toLowerCase();
  try { host = new URL(/^https?:\/\//.test(host) ? host : "https://" + host).hostname; } catch { return { kind: "unknown", reasons: [["warn", "That website address could not be read."]] }; }
  host = host.replace(/^www\./, "");
  if (/^http:\/\//i.test(raw)) reasons.push(["warn", "The link starts with http:// (not secure): payment details can be read in transit."]);
  for (const [brand, domains] of Object.entries(OFFICIAL))
    if (domains.some(d => host === d || host.endsWith("." + d))) return { kind: "official", store: NAMES[brand], reasons: [["good", `${host} is ${NAMES[brand]}'s official website.`], ...reasons] };
  const labels = host.split("."), ending = labels.at(-1);
  const main = labels.length >= 3 && ["co", "org", "net"].includes(labels.at(-2)) ? labels.at(-3) : labels.at(-2) || labels[0];
  // name parts of the address: "amaz0n-deals.shop" -> amazon, deals ("pineapplefarm" is not "apple": whole parts only)
  const parts = labels.slice(0, -1).flatMap(l => l.split("-")).map(unswap).filter(Boolean);
  let fake = null;
  for (const brand of Object.keys(OFFICIAL)) {
    if (parts.some(t => t === brand || (brand.length >= 6 && t.startsWith(brand)))) { fake = [brand, `uses the name "${NAMES[brand]}" but is not ${NAMES[brand]}'s website (the real one is ${OFFICIAL[brand][0]})`]; break; }
    const m = unswap(main), dist = levenshtein(m, brand);
    if (m !== brand && ((brand.length >= 7 && dist <= 2) || (brand.length >= 5 && dist === 1))) { fake = [brand, `is spelled like "${NAMES[brand]}" (${main} vs ${brand}): a lookalike website`]; break; }
  }
  if (fake) reasons.unshift(["bad", `${host} ${fake[1]}. This is how fake store websites trick buyers.`]);
  if (RISKY_ENDINGS.includes(ending)) reasons.push(["warn", `The address ends in ".${ending}", an ending that costs very little and is common on short-lived scam shops.`]);
  if (!fake) reasons.push(["warn", `${host} is not a store DupeLens knows. Check its reviews, contact address and return policy before paying.`]);
  return { kind: fake ? "fake" : RISKY_ENDINGS.includes(ending) ? "risky" : "unknown", store: fake ? NAMES[fake[0]] : null, reasons };
}
