(() => {
  let repository = window.RESEARCH_REPOSITORY || "";
  const host = location.hostname;
  if (!repository && host.endsWith(".github.io")) {
    const owner = host.slice(0, -10);
    const project = location.pathname.split("/").filter(Boolean)[0];
    repository = `https://github.com/${owner}/${project || host}`;
  }
  if (repository) {
    repository = repository.replace(/\/$/, "");
    document.querySelectorAll("[data-repo-path]").forEach(link => {
      const path = link.dataset.repoPath;
      link.href = `${repository}/${path.endsWith("/") ? "tree" : "blob"}/main/${path}`;
    });
  }
  const dialog = document.getElementById("explorer-dialog");
  const frame = document.getElementById("explorer-frame");
  const title = document.getElementById("explorer-title");
  const pageLink = document.getElementById("explorer-new");
  document.querySelectorAll("[data-explorer]").forEach(link => {
    link.addEventListener("click", event => {
      if (!dialog.showModal || event.ctrlKey || event.metaKey || event.shiftKey) return;
      event.preventDefault();
      title.textContent = link.dataset.explorer;
      frame.title = link.dataset.explorer;
      frame.src = link.href;
      pageLink.href = link.href;
      document.body.classList.add("dialog-open");
      document.querySelectorAll("video").forEach(video => video.pause());
      dialog.showModal();
    });
  });
  document.getElementById("explorer-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    frame.src = "about:blank";
    document.body.classList.remove("dialog-open");
  });
  dialog.addEventListener("click", event => {
    if (event.target === dialog) {
      const r = dialog.getBoundingClientRect();
      if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
    }
  });
  document.querySelectorAll("video").forEach(video => {
    video.addEventListener("play", () => {
      document.querySelectorAll("video").forEach(other => { if (other !== video) other.pause(); });
    });
  });
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (!entry.isIntersecting) entry.target.pause();
    }), {threshold: 0.05});
    document.querySelectorAll("video").forEach(video => observer.observe(video));
  }
})();
