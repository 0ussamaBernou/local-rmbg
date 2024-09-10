<script lang="ts">
    import "../app.css";
    import {ModeWatcher} from "mode-watcher";
    import TheHeader from "./TheHeader.svelte";
    import TheFooter from "./TheFooter.svelte";
</script>

<ModeWatcher />

<div class="flex flex-col justify-between items-stretch min-h-screen w-full">
    <TheHeader />
        <slot />
    <TheFooter />
</div>
