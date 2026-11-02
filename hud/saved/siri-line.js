// The HUD orb's Siri-style line, saved Nov 1 from orb-glass.html's Bubble look (frozen beside this file as
// bubble-v1.html) so it can be put back exactly as it is. Three harmonics, each a row of lobes mirrored about
// the middle, lit in the 60-30-10 colors and outlined in a liquid-metal gradient, clipped to inside the orb.
//
//   <script src="siri-line.js"></script>
//   const line = SiriLine();
//   // every frame, with wave eased toward the state's value at 3.5/s and level toward the voice at 14/s:
//   line.step(dt, wave, level);
//   line.draw(g, ox, oy, R * .72, R, [navyRGB, museRGB, modelRGB], devicePixelRatio);
//
// wave is idle .05, listening .45, thinking .12, speaking 1; level is the voice, 0 to 1. R is the orb's
// radius in canvas pixels, and the line reaches .92 of R * .72 each side of the middle. Colors are [r, g, b].
(function () {
  const HARMONICS = [[1, 1, 0], [.75, 1.45, 2.1], [.6, 1.9, 4.4]];
  const rgba = (c, a) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

  window.SiriLine = function () {
    const line = { phase: 0, wave: 0, level: 0, HARMONICS };

    line.step = (dt, wave, level) => {
      line.wave = wave;
      line.level = level;
      line.phase += dt * (1.5 + 7 * wave * (0.4 + level));
    };

    // one harmonic's lobes: their half-height at each point along the line
    line.lobes = (ox, half, amp, fr, off) => {
      const pts = [];
      for (let j = 0; j <= 120; j++) {
        const u = -2 + 4 * j / 120, att = Math.pow(4 / (4 + u ** 4), 4);
        pts.push([ox + u / 2 * half, Math.abs(amp * att * Math.sin(fr * 3.2 * u - line.phase * (.7 + .35 * fr) + off))]);
      }
      return pts;
    };

    line.draw = (g, ox, oy, Rd, R, colors, DPR) => {
      const half = Rd * .92, amp = Rd * .42 * (.03 + line.wave * (.25 + .75 * line.level));
      const chrome = g.createLinearGradient(ox - half, 0, ox + half, 0);
      chrome.addColorStop(0, 'rgba(255,255,255,0)');
      chrome.addColorStop(.18, 'rgba(215,228,255,.9)');
      chrome.addColorStop(.42, 'rgba(255,255,255,1)');
      chrome.addColorStop(.6, 'rgba(140,165,225,.75)');
      chrome.addColorStop(.8, 'rgba(255,185,130,.9)');
      chrome.addColorStop(1, 'rgba(255,255,255,0)');
      g.save();
      // as it drew in the Bubble look, whatever the page left set
      g.globalAlpha = 1;
      g.globalCompositeOperation = 'source-over';
      g.lineCap = 'round';
      g.beginPath();
      g.arc(ox, oy, R * .94, 0, Math.PI * 2);
      g.clip();
      HARMONICS.forEach(([a, fr, off], i) => {
        const pts = line.lobes(ox, half, amp * a, fr, off);
        g.beginPath();
        g.moveTo(pts[0][0], oy);
        for (const [x, h] of pts) g.lineTo(x, oy - h);
        for (let j = pts.length - 1; j >= 0; j--) g.lineTo(pts[j][0], oy + pts[j][1]);
        g.closePath();
        g.globalCompositeOperation = 'lighter';
        g.fillStyle = rgba(colors[i], .42);
        g.fill();
        g.globalCompositeOperation = 'source-over';
        g.strokeStyle = chrome;
        g.lineWidth = 1.1 * DPR;
        g.stroke();
      });
      g.lineWidth = 1.4 * DPR;
      g.beginPath();
      g.moveTo(ox - half, oy);
      g.lineTo(ox + half, oy);
      g.stroke();
      g.restore();
    };

    return line;
  };
})();
