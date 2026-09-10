import { Palette } from "lucide-react";

export default function ThemeControl() {
  return (
    <label className="theme-control">
      <Palette size={15} />
      <span>Theme</span>
      <select value="default" aria-label="Interface theme" onChange={() => undefined}>
        <option value="default">Default</option>
      </select>
    </label>
  );
}
