# Custom Emotes Folder

This folder is for adding your own custom emotes beyond the default 4 Clash Royale emotes!

## How to Add New Emotes

### Step 1: Add Screenshots
Put reference images in the `screenshots/` folder with this naming format:
```
screenshots/
├── emote_name_1.png
├── emote_name_2.jpg
├── another_emote.png
└── cool_gesture.jpg
```

**Naming Rules:**
- Use lowercase with underscores (e.g., `thumbs_up`, `peace_sign`, `dabbing`)
- Supported formats: `.png`, `.jpg`, `.jpeg`
- The filename (without extension) becomes the emote name
- Examples:
  - `thumbs_up.png` → Emote name: "Thumbs Up"
  - `peace_sign.jpg` → Emote name: "Peace Sign"
  - `victory_dance.png` → Emote name: "Victory Dance"

### Step 2: Add Sounds (Optional)
Put matching audio files in the `sounds/` folder:
```
sounds/
├── emote_name_1.mp3
├── emote_name_2.wav
├── another_emote.mp3
└── cool_gesture.mp3
```

**Sound Rules:**
- Same filename as the screenshot (but with audio extension)
- Supported formats: `.mp3`, `.wav`, `.ogg`
- If no sound file exists, the emote will be silent

### Step 3: Collect Training Data
Once you've added screenshots, run the data collection tool to record yourself performing the emote:
```bash
python scripts/record_multimodal.py --emote "Thumbs Up"
```

You'll need to perform the gesture 20-50 times for good accuracy.

### Step 4: Retrain Model
After collecting data, retrain the model:
```bash
python train/train_multimodal.py --include-custom-emotes
```

**NOTE:** Training requires a powerful GPU. If your current laptop isn't powerful enough:
- Option A: Use Google Colab (free GPU)
- Option B: Use a cloud service (AWS, Azure, etc.)
- Option C: Ask someone with a gaming PC to run training

---

## Example Custom Emotes

### Gaming Gestures
- `gg_gesture` - Good game gesture
- `rage_quit` - Frustrated throwing arms
- `victory_dance` - Celebration dance
- `thumbs_up` - Approval gesture
- `thumbs_down` - Disapproval gesture

### Social Emotes
- `wave_hello` - Friendly wave
- `peace_sign` - Peace/victory sign
- `facepalm` - Hand on face
- `shrug` - Shoulder shrug
- `thinking` - Hand on chin

### Clash Royale Specific
- `elixir_leak` - Panic gesture
- `king_tower_activate` - Celebration
- `bm_emote` - Taunt gesture
- `good_luck` - Friendly gesture
- `thanks` - Gratitude gesture

---

## Current Custom Emotes

(The system will auto-populate this list when you add emotes)

---

## Tips for Good Detection

1. **Good Lighting:** Make sure your face and hands are well-lit
2. **Clear Background:** Solid color backgrounds work best
3. **Full Body Visible:** Keep your upper body in frame
4. **Consistent Pose:** Perform the emote the same way each time
5. **Hold for 2-3 Seconds:** Don't rush the gesture
6. **Multiple Angles:** Record from slightly different positions

---

## Troubleshooting

**Q: My emote isn't being detected**
- A: Collect more training samples (aim for 50+)
- A: Make sure the pose is distinct from other emotes
- A: Check that your lighting and camera position are good

**Q: How do I delete an emote?**
- A: Remove the screenshot and sound files
- A: Delete training data from `data/multimodal/`
- A: Retrain the model

**Q: Can I have more than 10 custom emotes?**
- A: Yes! The system supports unlimited emotes, but:
  - More emotes = longer training time
  - More emotes = need more data per emote
  - More emotes = slightly lower accuracy

---

**Need help?** Check the main README.md or open an issue on GitHub!
