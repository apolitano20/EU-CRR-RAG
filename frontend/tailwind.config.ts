import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        eu: {
          blue: "#003399",
          "blue-dark": "#002277",
          gold: "#FFCC00",
        },
      },
    },
  },
  plugins: [],
};

export default config;
