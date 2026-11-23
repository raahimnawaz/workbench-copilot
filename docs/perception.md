# Perception

How one photo of the breadboard becomes a netlist checked against the firmware, and how the desk camera finds things in the arm's frame. Back to the [README](../README.md).

## Reading the breadboard

`read_breadboard` turns one photo into each jumper's two holes, the netlist, and every place it disagrees with `pins.h`:

1. **Tags.** The breadboard lies on a printed carrier sheet with four AprilTags (tag36h11, ids 10–13, 20 mm). Their corners map the photo onto the sheet in millimeters, so a bumped camera or a refocus doesn't move the hole grid, and there's no camera calibration to keep.
2. **Top-down view.** Code cuts out the board area at 5.5 px/mm, about 970 × 360 px. That's under the 1008 px SAM works at, so thin wires keep their detail, and SAM's masks come back in board millimeters.
3. **SAM 3.1 and the hole grid.** SAM segments `wire` in the view: one plain noun per request, since "jumper wire" matches nothing. Meanwhile code finds where the board really lies: it shifts, turns and scales the hole grid, within half a pitch, until the predicted holes sit on the darkest spots. The board's top stands about 10 mm above the sheet, so the camera sees it about 2% larger than life, and only the right scale lines up all 63 rows.
4. **Wires.** Each mask is thinned to its center line and followed out to both tips. Each tip snaps to the nearest hole within 1.3 mm, and a tip farther from every hole is flagged.
5. **Netlist and `pins.h`.** Each row's a–e and f–j strips and each rail is a node, wires join nodes, and the ESP32's and the LED's pins sit in holes given in `config/bench.yaml`. Plain code follows each `pins.h` signal from its part's pin to the GPIO its net reaches, and reports any mismatch, its wire and hole, and the line to fix.

**The demo board** is an ESP32 DevKit lighting an LED: one jumper from the DevKit's header to a row, 330 Ω from that row to the LED's anode, and the LED's cathode to ground. The DevKit straddles the center channel, which leaves only column a free beside its pins, so the board carries one signal wire. Its firmware keeps the LED on and says `led on` over serial: watch it with `.venv/bin/pio device monitor -b 115200`. A force sensor board on the same jumper (`hardware/demo_sensor.kicad_sch`) is what `read_sensor` reads. The reader reads flat pre-cut jumpers: loose leads stand about 15 mm off the board, and their tips look up to 4 mm from where they go in.

```bash
python scripts/make_carrier.py          # .wbc/vision/carrier.pdf: print at 100%, check the 100 mm bar
python scripts/read_board.py cameras    # free: which cameras macOS and OpenCV see
python scripts/read_board.py check      # free: the tags, and the fitted hole grid drawn over the board
python scripts/read_board.py read       # $0.0025: the whole read, drawn in .wbc/vision/board-<time>.png
```

- Run them from Terminal.app, which macOS lets use the camera. Each takes a still from the camera in `bench.yaml`, or reads one with `--photo`. The C920 refocuses every time it's opened, so the capture waits 1.5 s, then keeps the sharpest of the next 1.5 s of frames.
- `config/bench.yaml` holds the camera, the carrier's tags, each board's place on the sheet, where each module's headers start, and which part pin each `pins.h` signal drives. `config/parts.yaml` holds the breadboard's geometry and the module pinouts. Change those and print the carrier again; never edit the sheet.
- Each read checks its own photo: a blurred top-down view (sharpness under 150, where good reads measured 389–556) or glare that blows out more than 1% of it is said among the read's problems, with what to change ("move the lamp"), and the read goes ahead.
- Each read saves the drawing, the top-down view and the result, with SAM's whole reply, as `.wbc/vision/board-<time>.*`.

## Finding things on the desk

A second C920 looks straight down at the desk from about 71 cm and sees the arm's whole reach, about 97 × 55 cm at 0.5 mm per pixel. `wbc/perception/desk.py` turns what it sees into the arm's frame, so the co-pilot can find "the cup" or "the screwdriver" and know where it is.

1. **Tags, touched once.** AprilTags (tag36h11, ids 20–25, 50 mm) lie flat around the arm's reach. The carrier's 20 mm tags can be touched too, so the arm over a few of the desk's still leaves enough in view. With the motors off, you put the gripper's tips on each tag's center, and forward kinematics records where it is in the arm's frame. Every photo is mapped through the tags it shows, so a bumped camera changes nothing.
2. **Top-down view and SAM 3.1.** Code cuts out the table around the arm at 1 px/mm and asks SAM for the thing by name.
3. **Height.** Something standing up looks further from the point under the camera than it is. The tags also place the camera itself, so given a thing's height, its outline's near and far edges put its middle back where it is: a cup's middle lands within 3 mm.

```bash
python scripts/desk_camera.py tags                                    # .wbc/vision/desk-tags.pdf: print at 100%
python scripts/desk_camera.py --port /dev/cu.usbmodemXXXX touch       # motors off: touch each tag's center
python scripts/desk_camera.py check                                   # free: how well the tags agree
python scripts/desk_camera.py table                                   # free: the desk's height under the arm, from the touches
python scripts/desk_camera.py locate cup --height 95                  # $0.0025: where the cup is in the arm's frame
```

`config/desk.yaml` holds the camera, the tags, the area SAM sees and the focal length used to place the camera.
