//+------------------------------------------------------------------+
//|  ScalpRL_M1_Render.mq5 — comité RL OR M1 hébergé sur Render       |
//|  Même robot que ScalpRL_Trend_M1, mais le cerveau est sur Internet |
//|  (render_m1/app.py) au lieu du service Python du PC.               |
//|  MT5 : Outils > Options > Expert Advisors > cocher                 |
//|  « Autoriser WebRequest pour les URL listées » + ajouter InpURL.   |
//|  MODE FANTÔME (par défaut) : demande chaque minute la décision à   |
//|  Render en suivant la position du robot actuel (magique 260501),   |
//|  l'écrit dans le journal (onglet Experts) mais ne passe AUCUN      |
//|  ordre. Pour trader : InpShadow = false (et arrêter l'autre robot).|
//|  FILTRE VAGUES (v1.10, idée de l'utilisateur 09/11/2026) : dans    |
//|  la tendance, le prix fait des vagues vertes (jusqu'à un pic) et   |
//|  rouges (jusqu'à un creux). L'EA refuse un BUY tant que la vague   |
//|  en cours est rouge (creux pas confirmé par un rebond de X ATR M1) |
//|  et un SELL tant qu'elle est verte. Option : refus trop près du    |
//|  pic / du creux. Les sorties décidées par l'IA ne changent pas.    |
//|  Le journal affiche chaque minute la tendance vue par l'IA        |
//|  (renvoyée par Render) et la vague en cours, filtre actif ou non.  |
//+------------------------------------------------------------------+
#property copyright "GoldStark"
#property version   "1.10"
#property strict
#include <Trade\Trade.mqh>

input string InpURL            = "https://scalprl-m1.onrender.com"; // Adresse du service Render (sans / à la fin)
input string InpToken          = "";        // Mot de passe SCALPRL_TOKEN (Render > Environment)
input bool   InpShadow         = true;      // FANTÔME : décisions seulement, aucun ordre
input long   InpShadowMagic    = 260501;    // Fantôme : position suivie = celle du robot actuel ScalpRL_Trend_M1
input double InpLots           = 0.01;      // Lot (mode réel)
input long   InpMagic          = 260502;    // Numéro magique de CE robot (mode réel)
input int    InpHistoryDays    = 130;       // Jours d'historique M1 envoyés à Render (~110 000 minutes utiles)
input int    InpTimeoutSec     = 50;        // Attente max d'une réponse de Render (s) — offre gratuite : ~10-20 s
input double InpSafetySL       = 25.0;      // SL de sécurité ($ de prix) — 0 = aucun
input int    InpMaxHoldMin     = 60;        // Durée max d'un trade en minutes (60, comme à l'entraînement)
input double InpMaxSpread      = 1.0;       // Pas d'entrée si spread > X $
input int    InpCloseBeforeEnd = 5;         // Fermer si la séance se termine dans moins de N min
input bool   InpWaveFilter     = false;     // Filtre vagues : BUY seulement en vague verte, SELL seulement en vague rouge
input double InpWaveATR        = 3.0;       // Taille mini d'une vague pour confirmer un pic / un creux (en ATR M1)
input double InpLateWaveATR    = 0;         // Pas d'entrée si la vague a déjà fait plus de X ATR M1 (0 = non)
input int    InpPeakZoneMin    = 0;         // Zone du pic : fourchette des N dernières minutes (0 = non)
input double InpPeakZonePct    = 80;        // Pas de BUY au-dessus de X % de cette fourchette (SELL : sous 100 - X %)

CTrade   trade;
datetime g_lastBar = 0, g_reqBar = 0;
long     g_reqId = 0, g_gmtOff = 0;
bool     g_histSent = false;
int      g_histTries = 0;
uint     g_lastHistTry = 0;
#define  CHUNK 15000                         // minutes M1 par envoi d'historique (~0,7 Mo)
#define  WAVE_BARS 3000                      // minutes M1 relues pour suivre les vagues (~2 jours)

//--- outils ---------------------------------------------------------
long ServerToUTC(datetime t) { return (long)t - g_gmtOff; }
long PosMagic() { return InpShadow ? InpShadowMagic : InpMagic; }

int CurrentPos(ulong &ticket, double &openPrice, datetime &openTime)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(!PositionSelectByTicket(tk)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != PosMagic()) continue;
      ticket = tk; openPrice = PositionGetDouble(POSITION_PRICE_OPEN); openTime = (datetime)PositionGetInteger(POSITION_TIME);
      return PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
   }
   return 0;
}

int TradesToday(datetime barTime)
{
   long utc = ServerToUTC(barTime);
   datetime dayStart = (datetime)((utc / 86400) * 86400 + g_gmtOff);
   if(!HistorySelect(dayStart, TimeCurrent() + 60)) return 0;
   int n = 0;
   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
   {
      ulong d = HistoryDealGetTicket(i);
      if(HistoryDealGetString(d, DEAL_SYMBOL) == _Symbol && HistoryDealGetInteger(d, DEAL_MAGIC) == PosMagic()
         && HistoryDealGetInteger(d, DEAL_ENTRY) == DEAL_ENTRY_IN) n++;
   }
   return n;
}

int MinutesToSessionEnd()
{
   MqlDateTime dt; datetime now = TimeTradeServer(); TimeToStruct(now, dt);
   datetime from, to; long sod = (long)now % 86400;
   for(uint s = 0; SymbolInfoSessionTrade(_Symbol, (ENUM_DAY_OF_WEEK)dt.day_of_week, s, from, to); s++)
   {
      long f = (long)from % 86400, t = (long)to; if(t > 86400) t = 86400;
      if(sod >= f && sod < t)
      {
         if(t >= 86400) return 100000;   // séance qui continue après minuit : on ne coupe pas
         return (int)((t - sod) / 60);
      }
   }
   return 0;   // hors séance
}

string BarLine(const MqlRates &r)
{
   return StringFormat("%I64d,%.2f,%.2f,%.2f,%.2f,%.3f,%I64d\n", ServerToUTC(r.time), r.open, r.high, r.low, r.close,
                       r.spread * _Point, r.tick_volume);
}

// POST vers Render ; renvoie le texte de la réponse (code HTTP dans code, -1 si pas de connexion)
string Http(string path, string body, int &code)
{
   char data[], res[]; string rh;
   string hd = "Content-Type: text/plain\r\n";
   if(InpToken != "") hd += "X-Token: " + InpToken + "\r\n";
   int n = StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(data, MathMax(n - 1, 0));                     // sans le zéro final
   ResetLastError();
   code = WebRequest("POST", InpURL + path, hd, InpTimeoutSec * 1000, data, res, rh);
   if(code == -1)
   {
      int e = GetLastError();
      if(e == 4014) PrintFormat("WebRequest refusé : ajoutez %s dans Outils > Options > Expert Advisors (Autoriser WebRequest)", InpURL);
      else PrintFormat("Render injoignable (erreur %d) : connexion Internet ? service en démarrage ?", e);
      return "";
   }
   string s = CharArrayToString(res, 0, WHOLE_ARRAY, CP_UTF8);
   StringTrimLeft(s); StringTrimRight(s);
   return s;
}

//--- filtre vagues (même calcul que research/scalp_m1_swing_test.py) --
// Bougies M1 FERMÉES. ATR M1 = moyenne exponentielle (1/14) du vrai range, comme les informations de l'IA.
// Zigzag : vague verte tant que le prix ne recule pas de InpWaveATR x ATR sous le plus haut de la vague
// (alors pic confirmé -> vague rouge), vague rouge tant qu'il ne remonte pas de InpWaveATR x ATR au-dessus du plus bas.
int    g_wDir = 0;                                   // vague en cours : 1 verte, -1 rouge, 0 inconnue
double g_wExt = 0, g_wStart = 0, g_wAtr = 0, g_wClose = 0, g_wPos = 0.5;

// Calcule la vague en cours (et la place du prix dans la fourchette InpPeakZoneMin) sur les bougies M1 fermées.
bool WaveState()
{
   g_wDir = 0;
   MqlRates r[]; ArraySetAsSeries(r, false);
   int n = CopyRates(_Symbol, PERIOD_M1, 1, WAVE_BARS, r);
   if(n < 200) return false;
   double atr = 0, ext = r[0].high, start = r[0].low;
   int d = 1;
   for(int t = 0; t < n; t++)
   {
      double tr = r[t].high - r[t].low;
      if(t > 0) tr = MathMax(tr, MathMax(MathAbs(r[t].high - r[t - 1].close), MathAbs(r[t].low - r[t - 1].close)));
      atr = (t == 0) ? tr : atr + (tr - atr) / 14.0;
      double a = MathMax(atr, 1e-6);
      if(t > 0 && r[t].time - r[t - 1].time > 15 * 60) { d = 1; ext = r[t].high; start = r[t].low; }   // pause / trou : on repart
      if(d == 1)
      {
         if(r[t].high > ext) ext = r[t].high;
         else if(r[t].low < ext - InpWaveATR * a) { d = -1; start = ext; ext = r[t].low; }   // pic confirmé -> vague rouge
      }
      else
      {
         if(r[t].low < ext) ext = r[t].low;
         else if(r[t].high > ext + InpWaveATR * a) { d = 1; start = ext; ext = r[t].high; }  // creux confirmé -> vague verte
      }
   }
   g_wDir = d; g_wExt = ext; g_wStart = start; g_wAtr = MathMax(atr, 1e-6); g_wClose = r[n - 1].close; g_wPos = 0.5;
   if(InpPeakZoneMin > 0 && n >= InpPeakZoneMin)
   {
      double hh = r[n - 1].high, ll = r[n - 1].low;
      for(int i = n - InpPeakZoneMin; i < n; i++) { hh = MathMax(hh, r[i].high); ll = MathMin(ll, r[i].low); }
      g_wPos = (hh - ll > 1e-9) ? (g_wClose - ll) / (hh - ll) : 0.5;
   }
   return true;
}

// Texte de la vague en cours pour le journal
string WaveText()
{
   if(g_wDir == 0) return "vague ?";
   double leg = (g_wClose - g_wStart) / g_wAtr;
   if(g_wDir == 1)
      return StringFormat("vague VERTE depuis le creux %.2f (%+.1f ATR, pic confirmé si retour à %.2f)", g_wStart, leg, g_wExt - InpWaveATR * g_wAtr);
   return StringFormat("vague ROUGE depuis le pic %.2f (%+.1f ATR, creux confirmé si remontée à %.2f)", g_wStart, leg, g_wExt + InpWaveATR * g_wAtr);
}

// true si l'entrée « target » (1 = BUY, -1 = SELL) doit être refusée (WaveState() doit avoir été appelé juste avant)
bool WaveBlock(int target, string &why)
{
   if(g_wDir == 0) return false;                                  // pas assez d'historique : pas de filtre
   double leg = (g_wClose - g_wStart) / g_wAtr;
   if(InpWaveFilter)
   {
      if(target == 1 && g_wDir != 1)
      { why = StringFormat("vague rouge en cours (plus bas %.2f, creux pas confirmé : il faut remonter à %.2f)", g_wExt, g_wExt + InpWaveATR * g_wAtr); return true; }
      if(target == -1 && g_wDir != -1)
      { why = StringFormat("vague verte en cours (plus haut %.2f, pic pas confirmé : il faut redescendre à %.2f)", g_wExt, g_wExt - InpWaveATR * g_wAtr); return true; }
      if(InpLateWaveATR > 0 && target * leg > InpLateWaveATR)
      { why = StringFormat("vague déjà longue (%.1f ATR depuis %.2f)", MathAbs(leg), g_wStart); return true; }
   }
   if(InpPeakZoneMin > 0)
   {
      if(target == 1 && g_wPos > InpPeakZonePct / 100.0)
      { why = StringFormat("trop près du pic (%.0f %% de la fourchette des %d dernières minutes)", g_wPos * 100, InpPeakZoneMin); return true; }
      if(target == -1 && g_wPos < 1 - InpPeakZonePct / 100.0)
      { why = StringFormat("trop près du creux (%.0f %% de la fourchette des %d dernières minutes)", g_wPos * 100, InpPeakZoneMin); return true; }
   }
   return false;
}

//--- historique -----------------------------------------------------
bool SendHistory()
{
   if(g_lastHistTry != 0 && GetTickCount() - g_lastHistTry < 15000) return false;
   g_lastHistTry = GetTickCount(); g_histTries++;
   MqlRates r[]; ArraySetAsSeries(r, false);
   datetime now = (datetime)SymbolInfoInteger(_Symbol, SYMBOL_TIME);   // dernier tick de l'OR
   if(now == 0) now = TimeCurrent();
   datetime from = now - (datetime)InpHistoryDays * 86400;
   int n = CopyRates(_Symbol, PERIOD_M1, from, now, r);
   if(n > 1) n--;                                                       // la dernière minute n'est pas encore fermée
   bool fresh = n > 0 && now - r[n - 1].time < 30 * 60;
   bool deep  = n > 0 && r[0].time - from < 10 * 86400;
   bool sync  = (bool)SeriesInfoInteger(_Symbol, PERIOD_M1, SERIES_SYNCHRONIZED);
   if(!(fresh && deep && (sync || g_histTries >= 6)) && g_histTries < 24)  // on laisse MT5 télécharger (jusqu'à ~6 min)
   {
      if(g_histTries % 3 == 1)
         PrintFormat("Historique M1 en cours de téléchargement par MT5 (%d minutes, du %s au %s)... nouvel essai",
                     n, n > 0 ? TimeToString(r[0].time) : "-", n > 0 ? TimeToString(r[n - 1].time) : "-");
      return false;
   }
   if(!fresh) { Print("Historique M1 pas à jour : vérifiez la connexion au serveur Deriv"); return false; }
   int parts = (n + CHUNK - 1) / CHUNK, code = 0;
   string ans = "";
   for(int k = 0; k < parts; k++)
   {
      string body = "time_utc,o,h,l,c,spread,volume\n";
      int i1 = MathMin(n, (k + 1) * CHUNK);
      for(int i = k * CHUNK; i < i1; i++) StringAdd(body, BarLine(r[i]));
      ans = Http(StringFormat("/history?part=%d&last=%d", k, k == parts - 1 ? 1 : 0), body, code);
      if(code != 200 || StringFind(ans, "ok") != 0)
      { PrintFormat("Envoi de l'historique à Render : échec (code %d, %s) -> nouvel essai dans 15 s", code, ans); return false; }
   }
   PrintFormat("Historique envoyé à Render : %d minutes M1 en %d morceaux (%s)", n, parts, ans);
   return true;
}

//--- cycle ----------------------------------------------------------
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(50);
   g_gmtOff = (long)MathRound((double)(TimeTradeServer() - TimeGMT()) / 1800.0) * 1800;
   g_histSent = false; g_histTries = 0; g_lastHistTry = 0;
   g_lastBar = iTime(_Symbol, PERIOD_M1, 0);
   EventSetMillisecondTimer(500);
   PrintFormat("ScalpRL_M1_Render prêt | %s | service %s | décalage serveur/UTC %d h",
               InpShadow ? StringFormat("MODE FANTÔME (aucun ordre, suit le robot magique %I64d)", InpShadowMagic) : "MODE RÉEL",
               InpURL, g_gmtOff / 3600);
   PrintFormat("Filtre vagues : %s | zone du pic : %s",
               InpWaveFilter ? StringFormat("ACTIF (vague de %.1f ATR M1%s)", InpWaveATR,
                                            InpLateWaveATR > 0 ? StringFormat(", pas d'entrée après %.1f ATR", InpLateWaveATR) : "") : "coupé",
               InpPeakZoneMin > 0 ? StringFormat("%d min, %.0f %%", InpPeakZoneMin, InpPeakZonePct) : "coupée");
   if(InpToken == "") Print("Attention : InpToken vide (le service Render refusera les requêtes si SCALPRL_TOKEN est défini)");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }

void OnTick() { if(g_histSent) CheckNewBar(); }

void OnTimer()
{
   if(!g_histSent) { g_histSent = SendHistory(); return; }
   CheckNewBar();
}

void CheckNewBar()
{
   datetime bar = iTime(_Symbol, PERIOD_M1, 0);
   if(bar == 0 || bar == g_lastBar) return;
   g_lastBar = bar;
   ulong tk; double op; datetime ot;
   int pos = CurrentPos(tk, op, ot);
   int age = 0; double upnl = 0;
   MqlTick q; SymbolInfoTick(_Symbol, q);
   double mid = (q.bid + q.ask) / 2;
   if(pos != 0)
   {
      age = (int)((bar - (ot - ot % 60)) / 60);
      upnl = pos * (mid - op);
      if(age >= InpMaxHoldMin)                       // durée max : le robot ferme avant de demander (fantôme : simulé)
      {
         if(!InpShadow) { trade.PositionClose(tk); Print("Durée max atteinte -> fermeture"); }
         pos = 0; age = 0; upnl = 0;
      }
   }
   // requête : état + 30 dernières minutes M1 fermées (même format que le robot du PC)
   g_reqId++;
   string body = StringFormat("%I64d,%I64d,%d,%.3f,%d,%d\n", g_reqId, ServerToUTC(bar), pos, upnl, age, TradesToday(bar));
   MqlRates r[]; ArraySetAsSeries(r, false);
   int n = CopyRates(_Symbol, PERIOD_M1, 1, 30, r);
   if(n <= 0) { PrintFormat("CopyRates M1 a échoué (%d)", GetLastError()); return; }
   for(int i = 0; i < n; i++) StringAdd(body, BarLine(r[i]));
   uint t0 = GetTickCount(); int code = 0;
   string ans = Http("/decide", body, code);
   if(code != 200) { if(code > 0) PrintFormat("Render a répondu %d (%s) -> aucune action sur cette bougie", code, ans); return; }
   if(ans == "need_history")
   { g_histSent = false; g_histTries = 0; g_lastHistTry = 0; Print("Render demande l'historique complet -> nouvel envoi"); return; }
   string f[];
   if(StringSplit(ans, ',', f) < 3 || f[0] == "error") { PrintFormat("Pas de décision de Render sur cette bougie (%s)", ans); return; }
   if(StringToInteger(f[0]) != g_reqId) return;
   g_reqBar = bar;
   string trend = "tendance IA ?";                                     // ancienne version du service : 3 champs seulement
   if(ArraySize(f) >= 5)
   {
      int tc = (int)StringToInteger(f[3]);
      trend = StringFormat("tendance IA %s (écart EMA50 M15 %+.2f ATR)", tc == 1 ? "HAUSSE" : (tc == -1 ? "BAISSE" : "AUCUNE"),
                           StringToDouble(f[4]));
   }
   Apply((int)StringToInteger(f[1]), f[2], GetTickCount() - t0, trend);
}

void Apply(int action, string votes, uint ms, string trend)
{
   int target = action == 1 ? 1 : (action == 2 ? -1 : 0);
   ulong tk; double op; datetime ot;
   int pos = CurrentPos(tk, op, ot);
   if(pos != 0 && (int)((g_reqBar - (ot - ot % 60)) / 60) >= InpMaxHoldMin) target = 0;
   WaveState();                                                        // vague en cours (journal + filtre)
   if(MinutesToSessionEnd() < InpCloseBeforeEnd) target = 0;          // fin de séance / week-end : à plat
   if(target != 0 && target != pos)                                    // nouvelle entrée : filtre vagues
   {
      string why = "";
      if(WaveBlock(target, why))
      {
         PrintFormat("%sFiltre vagues : %s du comité refusé -> %s", InpShadow ? "FANTÔME | " : "", target == 1 ? "BUY" : "SELL", why);
         target = 0;                                                   // pas d'entrée (si l'IA retournait sa position, elle sort quand même)
      }
   }
   string txt = target == 1 ? "BUY" : (target == -1 ? "SELL" : "A PLAT");
   string info = trend + " | " + WaveText();
   if(InpShadow)
   { PrintFormat("FANTÔME | Render : %s (votes %s, réponse %u ms) | %s | position du robot actuel %d", txt, votes, ms, info, pos); return; }
   PrintFormat("Comité Render : %s (votes %s, réponse %u ms) | %s | position %d", txt, votes, ms, info, pos);
   if(target == pos) return;
   if(pos != 0) trade.PositionClose(tk);
   if(target == 0) return;
   MqlTick q; SymbolInfoTick(_Symbol, q);
   if(q.ask - q.bid > InpMaxSpread) { PrintFormat("Spread %.2f $ trop élevé -> pas d'entrée", q.ask - q.bid); return; }
   double sl = 0;
   if(InpSafetySL > 0) sl = NormalizeDouble(target == 1 ? q.bid - InpSafetySL : q.ask + InpSafetySL, _Digits);
   bool ok = target == 1 ? trade.Buy(InpLots, _Symbol, 0, sl, 0, "ScalpRL_M1_Render")
                         : trade.Sell(InpLots, _Symbol, 0, sl, 0, "ScalpRL_M1_Render");
   if(!ok) PrintFormat("Ordre refusé : %d %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
}
//+------------------------------------------------------------------+
