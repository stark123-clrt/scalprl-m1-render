//+------------------------------------------------------------------+
//|  ScalpRL_Trend_M1.mq5 — comité RL OR M1 (suivi de tendance EMA50 M15) |
//|  Cerveau : service Python (scalp_rl_live/service.py) sur le PC.    |
//|  Échange par fichiers dans  Common\Files\scalp_rl  (aucune URL).   |
//|  L'EA : envoie l'état à chaque clôture M1, applique la décision,  |
//|  gère SL de sécurité, stop suiveur, durée max 60 min, pauses.      |
//+------------------------------------------------------------------+
#property copyright "GoldStark"
#property version   "1.00"
#property strict
#include <Trade\Trade.mqh>

input ENUM_TIMEFRAMES InpTF     = PERIOD_M1;      // Unité de décision du comité (M1)
input string InpBridge          = "scalp_rl_m1";     // Dossier d'échange du comité OR M1
input double InpLots            = 0.01;    // Lot
input long   InpMagic           = 260501;  // Numéro magique (différent des robots M5 / M15)
input int    InpHistoryDays     = 200;     // Jours d'historique M1 envoyés au démarrage (~6 mois)
input int    InpTimeoutSec      = 40;      // Attente max de la réponse du comité (s) — une décision par minute
input double InpSafetySL        = 25.0;    // SL de sécurité ($ de prix) — 0 = aucun
input bool   InpUseTrailing     = false;    // Stop suiveur
input double InpTrailStart      = 3.0;     // Suiveur : activé à +X $ de gain
input double InpTrailDist       = 2.0;     // Suiveur : distance au meilleur prix ($)
input bool   InpReentryConfirm  = false;    // Après une sortie SL/suiveur : réouverture seulement si bougie de confirmation
input int    InpConfirmBars     = 3;       // ... pendant N bougies après la sortie
input int    InpMaxHoldMin      = 60;      // Durée max d'un trade en minutes (60, comme à l'entraînement)
input double InpMaxSpread       = 1.0;     // Pas d'entrée si spread > X $
input int    InpCloseBeforeEnd  = 5;       // Fermer si la session de cotation se termine dans moins de N min
input int    InpCandleConfirm   = 0;       // Entrée seulement si les N dernières bougies vont majoritairement dans le sens du vote (0 = non, 2 ou 3)
input group "Protections contre les grosses pertes (à activer selon le test)"
input bool   InpNotWorking      = false;   // P2 Couper un trade qui ne marche pas
input int    InpNotWorkingBars  = 3;       //    ... après N bougies
input double InpNotWorkingATR   = 1.0;     //    ... s'il perd plus de X ATR
input bool   InpBreakEven       = false;   // P3 Break-even
input double InpBEStart         = 1.5;     //    ... dès +X $ de gain
input double InpBEOffset        = 0.1;     //    ... SL placé à entrée + X $
input bool   InpDoubtExit       = false;   // P4 Sortir dès qu'UN agent vote le sens opposé
input double InpSLATR           = 0.0;     // P5 SL = X ATR (0 = SL de sécurité fixe)
input bool   InpKnifeFilter     = false;   // P1 Pas d'entrée contre une grosse bougie
input double InpKnifeATR        = 1.5;     //    ... bougie plus grande que X ATR
input bool   InpRSIMomentum     = false;   // P6a Pas de BUY si RSI < 30 et en baisse (SELL : > 70 et en hausse)
input bool   InpRSIExtreme      = false;   // P6b Pas de BUY si RSI > 80 (SELL : RSI < 20)
input bool   InpADXFilter       = false;   // P7 Pas d'entrée contre une tendance forte (ADX)
input double InpADXLevel        = 25.0;    //    ... ADX au-dessus de

CTrade   trade;
string   DIR = "scalp_rl\\";
int      g_ps = 300;
datetime g_lastBar = 0, g_reqBar = 0, g_lastEAExitBar = 0;
long     g_reqId = 0;
bool     g_pending = false, g_lastExitByEA = false, g_histSent = false;
int      g_histTries = 0;
uint     g_reqTick = 0;
long     g_gmtOff = 0;
int      hATR = INVALID_HANDLE, hRSI = INVALID_HANDLE, hADX = INVALID_HANDLE;

double Buf(int h, int buffer, int shift)
{
   double v[]; if(CopyBuffer(h, buffer, shift, 1, v) != 1) return EMPTY_VALUE; return v[0];
}

// filtres d'entrée (P1, P6a, P6b, P7) sur la dernière bougie fermée
string EntryBlocked(int d)
{
   double o = iOpen(_Symbol, InpTF, 1), c = iClose(_Symbol, InpTF, 1);
   double hi = iHigh(_Symbol, InpTF, 1), lo = iLow(_Symbol, InpTF, 1), atr = Buf(hATR, 0, 1);
   int col = c > o ? 1 : (c < o ? -1 : 0);
   if(InpCandleConfirm > 0)
   {
      int same = 0;
      for(int k = 1; k <= InpCandleConfirm; k++)
      { double ok = iOpen(_Symbol, InpTF, k), ck = iClose(_Symbol, InpTF, k); if((ck > ok ? 1 : (ck < ok ? -1 : 0)) == d) same++; }
      if(2 * same <= InpCandleConfirm) return StringFormat("seulement %d bougie(s) sur %d dans le sens du vote", same, InpCandleConfirm);
   }
   if(InpKnifeFilter && atr != EMPTY_VALUE && hi - lo > InpKnifeATR * atr && col == -d) return "grosse bougie contraire (couteau)";
   double r = Buf(hRSI, 0, 1), rp = Buf(hRSI, 0, 2);
   if(InpRSIMomentum && r != EMPTY_VALUE && ((d == 1 && r < 30 && r < rp) || (d == -1 && r > 70 && r > rp))) return StringFormat("RSI %.0f encore en impulsion contraire", r);
   if(InpRSIExtreme && r != EMPTY_VALUE && ((d == 1 && r > 80) || (d == -1 && r < 20))) return StringFormat("RSI %.0f extrême", r);
   double adx = Buf(hADX, 0, 1), pdi = Buf(hADX, 1, 1), ndi = Buf(hADX, 2, 1);
   if(InpADXFilter && adx != EMPTY_VALUE && adx > InpADXLevel && ((d == 1 && ndi > pdi) || (d == -1 && pdi > ndi))) return StringFormat("tendance forte contraire (ADX %.0f)", adx);
   return "";
}

//--- outils ---------------------------------------------------------
long ServerToUTC(datetime t) { return (long)t - g_gmtOff; }

int CurrentPos(ulong &ticket, double &openPrice, datetime &openTime)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(!PositionSelectByTicket(tk)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
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
      if(HistoryDealGetString(d, DEAL_SYMBOL) == _Symbol && HistoryDealGetInteger(d, DEAL_MAGIC) == InpMagic
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

bool WriteBars(string name, int count, string header)
{
   MqlRates r[];
   ArraySetAsSeries(r, false);
   int n;
   if(name == "history.csv")
   {
      datetime now = (datetime)SymbolInfoInteger(_Symbol, SYMBOL_TIME);   // dernier tick de l'OR (pas des autres symboles)
      if(now == 0) now = TimeCurrent();
      datetime from = now - (datetime)InpHistoryDays * 86400;
      n = CopyRates(_Symbol, PERIOD_M1, from, now, r);
      if(n > 1) n--;                                            // la dernière minute n'est pas encore fermée
      bool fresh = n > 0 && now - r[n - 1].time < 30 * 60;     // (marché fermé : comparé au dernier tick de l'or)
      bool deep  = n > 0 && r[0].time - from < 10 * 86400;
      bool sync  = (bool)SeriesInfoInteger(_Symbol, PERIOD_M1, SERIES_SYNCHRONIZED);
      if(!(fresh && deep && (sync || g_histTries >= 6)) && g_histTries < 36)          // on attend le téléchargement (jusqu'à ~6 min)
      {
         if(g_histTries % 3 == 0)
            PrintFormat("Historique M1 en cours de téléchargement par MT5 (%d minutes, du %s au %s)... nouvel essai",
                        n, n > 0 ? TimeToString(r[0].time) : "-", n > 0 ? TimeToString(r[n - 1].time) : "-");
         return false;
      }
      if(!fresh) { Print("Historique M1 pas à jour : vérifiez la connexion au serveur Deriv"); return false; }
   }
   else
   {
      n = CopyRates(_Symbol, PERIOD_M1, 1, count, r);
   }
   if(n <= 0) { PrintFormat("CopyRates M1 a échoué (%d)", GetLastError()); return false; }
   int h = FileOpen(DIR + name + ".tmp", FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) { PrintFormat("Écriture impossible %s (%d)", name, GetLastError()); return false; }
   FileWriteString(h, header + "\n");
   for(int i = 0; i < n; i++)
      FileWriteString(h, StringFormat("%I64d,%.2f,%.2f,%.2f,%.2f,%.3f,%I64d\n", ServerToUTC(r[i].time), r[i].open, r[i].high,
                      r[i].low, r[i].close, r[i].spread * _Point, r[i].tick_volume));
   FileClose(h);
   if(!FileMove(DIR + name + ".tmp", FILE_COMMON, DIR + name, FILE_COMMON | FILE_REWRITE))
   { PrintFormat("Renommage %s impossible (%d)", name, GetLastError()); return false; }
   if(name == "history.csv") PrintFormat("Historique envoyé au comité : %d minutes M1", n);
   return true;
}

//--- cycle ----------------------------------------------------------
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(50);
   g_gmtOff = (long)MathRound((double)(TimeTradeServer() - TimeGMT()) / 1800.0) * 1800;
   hATR = iATR(_Symbol, InpTF, 14); hRSI = iRSI(_Symbol, InpTF, 14, PRICE_CLOSE); hADX = iADX(_Symbol, InpTF, 14);
   g_ps = PeriodSeconds(InpTF); DIR = InpBridge + "\\";
   FolderCreate(InpBridge, FILE_COMMON);
   FileDelete(DIR + "response.csv", FILE_COMMON);
   g_histSent = false; g_histTries = 0;
   g_lastBar = iTime(_Symbol, InpTF, 0);
   EventSetMillisecondTimer(500);
   SendHistory();
   PrintFormat("ScalpRL_Trend_M1 prêt | décalage serveur/UTC %d h | lancez scalp_rl_live\\start_service_m1.bat", g_gmtOff / 3600);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }

void SendHistory()
{
   static uint last = 0;
   if(g_histSent || (last != 0 && GetTickCount() - last < 10000)) return;
   last = GetTickCount(); g_histTries++;
   g_histSent = WriteBars("history.csv", 0, "time_utc,o,h,l,c,spread,volume");
}

void OnTick() { ManageTrailing(); if(g_histSent) CheckNewBar(); }

void OnTimer()
{
   if(FileIsExist(DIR + "need_history.flag", FILE_COMMON))           // le service a vu un trou dans l'historique
   { FileDelete(DIR + "need_history.flag", FILE_COMMON); g_histSent = false; g_histTries = 0; g_pending = false;
     Print("Le comité demande l'historique complet -> nouvel envoi"); }
   if(!g_histSent) { SendHistory(); return; }
   CheckNewBar(); ReadResponse();
}

void CheckNewBar()
{
   datetime bar = iTime(_Symbol, InpTF, 0);
   if(bar == 0 || bar == g_lastBar) return;
   g_lastBar = bar;
   ulong tk; double op; datetime ot;
   int pos = CurrentPos(tk, op, ot);
   int age = 0; double upnl = 0;
   MqlTick q; SymbolInfoTick(_Symbol, q);
   double mid = (q.bid + q.ask) / 2;
   if(pos != 0)
   {
      age = (int)((bar - (ot - ot % g_ps)) / g_ps);
      upnl = pos * (mid - op);
      if(age * g_ps / 60 >= InpMaxHoldMin) { trade.PositionClose(tk); Print("Durée max atteinte -> fermeture"); g_lastExitByEA = false; pos = 0; age = 0; upnl = 0; }
   }
   // requête : état + 30 dernières minutes M1
   g_reqId++;
   string head = StringFormat("%I64d,%I64d,%d,%.3f,%d,%d", g_reqId, ServerToUTC(bar), pos, upnl, age, TradesToday(bar));
   FileDelete(DIR + "response.csv", FILE_COMMON);
   if(WriteBars("request.csv", 30, head)) { g_pending = true; g_reqBar = bar; g_reqTick = GetTickCount(); }
}

void ReadResponse()
{
   if(!g_pending) return;
   if(GetTickCount() - g_reqTick > (uint)InpTimeoutSec * 1000)
   { g_pending = false; Print("Pas de réponse du comité (service Python lancé ?) -> aucune action sur cette bougie"); return; }
   if(!FileIsExist(DIR + "response.csv", FILE_COMMON)) return;
   int h = FileOpen(DIR + "response.csv", FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) return;
   string line = FileReadString(h); FileClose(h);
   string f[]; if(StringSplit(line, ',', f) < 5) return;
   if(StringToInteger(f[0]) != g_reqId) return;
   g_pending = false;
   FileDelete(DIR + "response.csv", FILE_COMMON);
   Apply((int)StringToInteger(f[1]), f[2] + f[3] + f[4]);
}

void Apply(int action, string votes)
{
   int target = action == 1 ? 1 : (action == 2 ? -1 : 0);
   ulong tk; double op; datetime ot;
   int pos = CurrentPos(tk, op, ot);
   if(pos != 0 && (int)((g_reqBar - (ot - ot % g_ps)) / g_ps) * g_ps / 60 >= InpMaxHoldMin) target = 0;
   if(MinutesToSessionEnd() < InpCloseBeforeEnd) target = 0;          // fin de séance / week-end : à plat
   if(pos != 0 && InpDoubtExit && StringFind(votes, pos == 1 ? "2" : "1") >= 0)
   { target = 0; Print("P4 : un agent vote le sens opposé -> sortie"); }
   if(pos != 0 && InpNotWorking)
   {
      int age = (int)((g_reqBar - (ot - ot % g_ps)) / g_ps); double atr = Buf(hATR, 0, 1);
      MqlTick qq; SymbolInfoTick(_Symbol, qq);
      double pl = pos == 1 ? qq.bid - op : op - qq.ask;
      if(atr != EMPTY_VALUE && age >= InpNotWorkingBars && pl < -InpNotWorkingATR * atr)
      { target = 0; PrintFormat("P2 : trade qui ne marche pas (%.2f $ après %d bougies) -> sortie", pl, age); }
   }
   PrintFormat("Comité : %s (votes %s) | position %d", target == 1 ? "BUY" : (target == -1 ? "SELL" : "A PLAT"), votes, pos);
   if(target == pos) return;
   if(pos != 0) { trade.PositionClose(tk); g_lastExitByEA = false; }
   if(target == 0) return;
   MqlTick q; SymbolInfoTick(_Symbol, q);
   if(q.ask - q.bid > InpMaxSpread) { PrintFormat("Spread %.2f $ trop élevé -> pas d'entrée", q.ask - q.bid); return; }
   string why = EntryBlocked(target);
   if(why != "") { Print("Entrée refusée : " + why); return; }
   if(InpReentryConfirm && g_lastExitByEA && (g_reqBar - g_lastEAExitBar) <= InpConfirmBars * g_ps)
   {
      double o1 = iOpen(_Symbol, InpTF, 1), c1 = iClose(_Symbol, InpTF, 1);
      if((target == 1 && c1 <= o1) || (target == -1 && c1 >= o1))
      { Print("Réouverture refusée : pas de bougie de confirmation"); return; }
   }
   double sl = 0, sld = InpSafetySL, atrE = Buf(hATR, 0, 1);
   if(InpSLATR > 0 && atrE != EMPTY_VALUE) sld = InpSLATR * atrE;
   if(sld > 0) sl = target == 1 ? q.bid - sld : q.ask + sld;
   sl = NormalizeDouble(sl, _Digits);
   bool ok = target == 1 ? trade.Buy(InpLots, _Symbol, 0, sl, 0, "ScalpRL_M1") : trade.Sell(InpLots, _Symbol, 0, sl, 0, "ScalpRL_M1");
   if(!ok) PrintFormat("Ordre refusé : %d %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
   else g_lastExitByEA = false;
}

void ManageTrailing()
{
   if(!InpUseTrailing && !InpBreakEven) return;
   ulong tk; double op; datetime ot;
   int pos = CurrentPos(tk, op, ot);
   if(pos == 0) return;
   MqlTick q; SymbolInfoTick(_Symbol, q);
   double gain = pos == 1 ? q.bid - op : op - q.ask;
   double nsl = 0;
   if(InpUseTrailing && gain >= InpTrailStart) nsl = pos == 1 ? q.bid - InpTrailDist : q.ask + InpTrailDist;
   if(InpBreakEven && gain >= InpBEStart)
   {
      double be = op + pos * InpBEOffset;
      if(nsl == 0 || (pos == 1 && be > nsl) || (pos == -1 && be < nsl)) nsl = be;
   }
   if(nsl == 0) return;
   double cur = PositionGetDouble(POSITION_SL), tp = PositionGetDouble(POSITION_TP);
   nsl = NormalizeDouble(nsl, _Digits);
   double minDist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(pos == 1 && (q.bid - nsl < minDist || (cur > 0 && nsl <= cur + 10 * _Point))) return;
   if(pos == -1 && (nsl - q.ask < minDist || (cur > 0 && nsl >= cur - 10 * _Point))) return;
   trade.PositionModify(tk, nsl, tp);
}

void OnTradeTransaction(const MqlTradeTransaction &tr, const MqlTradeRequest &rq, const MqlTradeResult &rs)
{
   if(tr.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(tr.deal)) return;
   if(HistoryDealGetInteger(tr.deal, DEAL_MAGIC) != InpMagic || HistoryDealGetInteger(tr.deal, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;
   if(HistoryDealGetInteger(tr.deal, DEAL_REASON) == DEAL_REASON_SL)
   { g_lastExitByEA = true; g_lastEAExitBar = iTime(_Symbol, InpTF, 0); Print("Sortie par SL / stop suiveur"); }
}
//+------------------------------------------------------------------+
