import { defineConfig } from 'vite';
export default defineConfig({base:'./',server:{host:'127.0.0.1',port:5178,strictPort:true,proxy:{'/api':process.env.NIGHTLAB_API_URL || 'http://127.0.0.1:8766'}},build:{outDir:'dist'}});
