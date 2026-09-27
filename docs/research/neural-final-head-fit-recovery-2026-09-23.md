# Final head fits after the interrupted session

The resumed session found 69 of 72 randomized/export non-student fits complete.
The remaining tasks were the DINO TCN, frozen MobileNet TCN and DINO transformer
for `export-rally-selection/split-20260923`. The old parent process was absent;
its exit status was not observed and is recorded as unknown.

The DINO TCN had reached epoch 43, with weights and predictions saved at epochs
5, 15 and 30. There was no optimizer checkpoint or completed fit receipt.
`recover-neural-final-head-fits.py` records a recovery plan, verifies all 69
completed fit gates, and preserves the partial directory and task log on the NAS.
It then repeats this one head from its unchanged registered seed. All arrays in
the six preserved checkpoint archives must exactly match the repeated run before
the other two heads proceed. The partial attempt is not counted as an additional
completed model or a selectable candidate.

The recovery invokes the unchanged registered fit queue. It changes no training
examples, labels, feature caches, seed, checkpoint bank, decoder or calibration
policy. The existing completed fits remain subject to their numerical and reuse
gates. Completion requires all 72 gates, the checkpoint parity receipt and actual
successful returns from the three newly launched commands. A supervisor records
the new process's observed exit separately from the unknown old exit.

The worker uses CPU 0–1 and two threads, at nice 10, with temporary files and
outputs on the NAS. It shares the existing maximum of three GPU workers and two
image workers. Each model starts only with at least 6 GiB Linux available memory,
3 GiB Windows free physical memory, 5 GiB free GPU memory and 20 GiB free on C:.
All 162 calibration choices still freeze before additional-video accuracy is
computed. No additional-video accuracy influenced this recovery decision.
