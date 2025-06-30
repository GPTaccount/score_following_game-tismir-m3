# Score Following as a Multi-Modal Reinforcement Learning Problem


### 修改自 https://github.com/CPJKU/score_following_game

可在 mac book air m1/m3 執行

### ⚠️ 關於 SoundFont 音色檔案

由於 GitHub 限制上傳超過 100MB 的檔案，本專案**壓縮** `grand-piano-YDP-20160804.sf2` 音色檔後上傳。

下載專案後需要解壓縮下列檔案：

score_following_game\sound_fonts\grand-piano-YDP-20160804.sf2.zip

### 測試命令

```
cd ~/project/score_following_game-tismir-m3

conda activate score_following_m3
```

```
python -m score_following_game.test_agent \
  --params score_following_game/models/reinforce-ScoreFollowingNetMSMDLCHSDeepDoLight-msmd_all_train-mutopia_lchs1_20190625_080050-florian/best_model.pt \
  --data_set score_following_game/data/test_sample \
  --piece Anonymous__lesgraces__lesgraces \
  --game_config score_following_game/game_configs/mutopia_lchs1.yaml \
  --agent_type rl
```


