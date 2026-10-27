% Draft captions, generated from the saved results. Numbers are read from the artifacts.

Fig. 1 (fig1_convergence). Global-train metric versus communication round for FedAvg
(E=5, persistent client optimizer) at K=2, IID and natural cross-session split; (a)
classification accuracy, (b) residual-regression standardized MSE (log scale; 1 = the
lookup). Bands span the trained init seeds (4601, 4602, 4603); a seed that did not train or
collapsed is drawn as its own dotted line. FL takes 8,400 gradient steps
(regression) and 12,000 (classification); the centralized references take
1,680 / 2,340 (60 epochs), and the
"centralized, matched steps" residual run 8,400 (300 epochs).

Fig. 2 (fig2_comparison). Centralized, FL and local-only models on the 28 held-out
runs, every partition condition. (a) Run accuracy by majority vote with Wilson 95%
intervals; (b) skill against the source/subject lookup with run-level bootstrap 95%
intervals (10,000 resamples), zero = the lookup. One point per init seed (FL,
centralized) or per client (local-only). The skill axis is clipped at −5; arrowheads
mark cut-off intervals and the hidden values are printed. Regression models that did not
train or collapsed are marked and labelled.

Fig. 3 (fig3_sensor_axis). FL and local-only performance against position overlap rho at
K=3 (contiguous position blocks, masked inputs, global 9-channel evaluation). Each FL
regression point is labelled with its training outcome; local-only shows mean ± std over
trained clients with each client faint behind it.

Fig. 4 (fig4_partitions). Training runs per client and (source, subject) for each
partition scheme at split seed 4601; row totals at right. The three rho layouts share
the IID K=3 run assignment and differ only in which positions each client holds.

Fig. 5 (fig5_run_scatter). Predicted versus actual walking speed per held-out run,
identity line; colour and marker by subject, filled = Test_2, hollow = 20251124. Init seed
4601 for centralized (residual) and FL K=2 IID; local-only K=2 IID, one panel per client.
